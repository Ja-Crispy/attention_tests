import torch
import torch.nn as nn
import torch.nn.functional as F # Added import for F.pad
from .vanilla_mha import MultiHeadAttention # Used for local and potentially global attention
from .combination_strategies import get_combination_strategy # Added for combination strategy

class AHCAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 chunk_size: int,
                 summarization_method_config: dict,
                 global_attention_method_config: dict,
                 combination_method_config: dict,
                 dropout_rate: float = 0.0,
                 **kwargs): # Added **kwargs for flexibility
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads # For local MHA
        self.chunk_size = chunk_size
        self.summarization_method_config = summarization_method_config
        self.global_attention_method_config = global_attention_method_config
        self.combination_method_config = combination_method_config
        self.dropout_rate = dropout_rate
        self.additional_config = kwargs

        # Validate chunk_size at initialization
        if self.chunk_size <= 0:
            raise ValueError("AHCAttention chunk_size must be positive.")
        
        model_max_length = kwargs.get('model_max_length') # Assuming you might pass this for validation
        if model_max_length and self.chunk_size > model_max_length:
            # This check might need refinement if very short sequences are padded up to chunk_size
            # but it's good for catching fundamental misconfigurations.
            raise ValueError(f"AHCAttention chunk_size ({self.chunk_size}) should not be greater than model_max_length ({model_max_length}) if model_max_length is provided.")

        # --- Configuration Validation ---
        # 1. Summarization Method Config
        summarization_type = self.summarization_method_config.get('type')
        allowed_summarization_types = ["S1_pool", "S2_linear_on_pool"] # Assuming None is not a valid type here, must be explicit
        if summarization_type not in allowed_summarization_types:
            raise ValueError(
                f"Invalid summarization_method_config['type']: {summarization_type}. "
                f"Must be one of {allowed_summarization_types}."
            )
        if summarization_type in ["S1_pool", "S2_linear_on_pool"]:
            pool_type = self.summarization_method_config.get('pool_type')
            allowed_pool_types = ["mean", "max"]
            if pool_type is None:
                raise ValueError(
                    f"summarization_method_config['pool_type'] must be specified for type '{summarization_type}'."
                )
            if pool_type not in allowed_pool_types:
                raise ValueError(
                    f"Invalid summarization_method_config['pool_type']: {pool_type}. "
                    f"Must be one of {allowed_pool_types} for type '{summarization_type}'."
                )

        # 2. Global Attention Method Config
        global_attention_type = self.global_attention_method_config.get('type')
        allowed_global_attention_types = ["G1_full_mha", None] # None means pass-through summary tokens
        if global_attention_type not in allowed_global_attention_types:
            raise ValueError(
                f"Invalid global_attention_method_config['type']: {global_attention_type}. "
                f"Must be one of {allowed_global_attention_types}."
            )

        # 3. Combination Method Config - Initialization handled by factory
        self.combination_strategy = get_combination_strategy(self.combination_method_config)
        # --- End Configuration Validation ---

        self.local_mha = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )

        self.global_mha = None
        if global_attention_type == "G1_full_mha":
            global_n_heads = self.global_attention_method_config.get('num_heads', self.n_heads)
            global_dropout_rate = self.global_attention_method_config.get('dropout_rate', self.dropout_rate)
            self.global_mha = MultiHeadAttention(
                d_model=self.d_model,
                n_heads=global_n_heads,
                dropout_rate=global_dropout_rate
            )
        elif global_attention_type is not None: # If type is specified, valid, but not G1 (and no other handlers)
             # This assumes G1_full_mha is the only implemented MHA-based global attention.
             # If other types were allowed by validation but don't have an nn.Module here, it's an issue.
            pass # Or raise NotImplementedError if new valid types are added to validation but not here

        self.s2_linear_projection = None
        if self.summarization_method_config.get('type') == "S2_linear_on_pool":
            self.s2_linear_projection = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def _process_input_mask(mask: torch.Tensor, target_seq_len: int, device: torch.device, target_dtype: torch.dtype = torch.float) -> torch.Tensor:
        """
        Processes an input attention mask into a canonical format suitable for chunking.
        Converts common padding mask formats (2D or a specific 4D)
        into a standard 4D tensor shape (batch_size, 1, 1, seq_len) with a float dtype,
        where 1.0 means keep and 0.0 means pad. This format is expected by the
        _chunk_input method for masks.
        Args:
            mask (torch.Tensor, optional): The input attention mask. Supported shapes:
                - None: Returns None.
                - 2D (batch_size, seq_len): Boolean or float/int. True/1 indicates keep.
                - 4D (batch_size, 1, 1, seq_len): Float. 1.0 indicates keep.
            target_seq_len (int): The expected sequence length for the output mask.
            device (torch.device): The target device for the output mask.
            target_dtype (torch.dtype, optional): The target data type for the output mask.
        Returns:
            torch.Tensor, optional: Processed mask (batch_size, 1, 1, seq_len), or None.
        Raises:
            ValueError: If mask has unsupported dimension or mismatched sequence length.
        """
        if mask is None:
            return None

        if mask.dtype == torch.bool:
            mask = mask.type(target_dtype)
        elif mask.dtype != target_dtype:
            mask = mask.type(target_dtype)

        if mask.ndim == 2:
            if mask.shape[1] != target_seq_len:
                raise ValueError(
                    f"Input 2D mask's sequence length ({mask.shape[1]}) does not match "
                    f"target sequence length ({target_seq_len})."
                )
            processed_mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.ndim == 4:
            if mask.shape[1] == 1 and mask.shape[2] == 1 and mask.shape[3] == target_seq_len:
                processed_mask = mask
            else:
                raise ValueError(
                    f"Input 4D mask has shape {mask.shape}. Expected (batch_size, 1, 1, {target_seq_len}) "
                    f"for this utility. Other 4D mask formats require custom handling."
                )
        else:
            raise ValueError(
                f"Unsupported mask dimension: {mask.ndim}. Expected 2D (batch_size, seq_len) "
                f"or 4D (batch_size, 1, 1, seq_len)."
            )

        if processed_mask.device != device:
            processed_mask = processed_mask.to(device)
            
        return processed_mask

    def _summarize_s1_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        pool_type = summarization_config.get('pool_type', 'mean')
        if pool_type == 'mean':
            summary_tokens = torch.mean(local_context, dim=2) # Pool over chunk_size dimension
        elif pool_type == 'max':
            summary_tokens = torch.max(local_context, dim=2).values # Pool over chunk_size dimension
        else:
            raise ValueError(f"Unsupported S1 pool_type: {pool_type}")
        return summary_tokens

    def _summarize_s2_linear_on_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        pooled_tokens = self._summarize_s1_pool(local_context, summarization_config)
        if self.s2_linear_projection is None:
            raise RuntimeError("S2 linear projection layer is not initialized. Check configuration for S2_linear_on_pool.")
        summary_tokens = self.s2_linear_projection(pooled_tokens)
        return summary_tokens

    def _chunk_input(self, input_tensor: torch.Tensor, input_tensor_mask: torch.Tensor = None):
        """
        Pads and reshapes the input tensor and its mask into chunks.
        Args:
            input_tensor (torch.Tensor): Shape (batch_size, original_seq_len, d_model).
            input_tensor_mask (torch.Tensor, optional): Shape (batch_size, 1, 1, original_seq_len).
                                                        Values: 0.0 for pad/mask, 1.0 for keep.
        Returns:
            Tuple[torch.Tensor, torch.Tensor | None, int, int]:
                - chunked_tensor (batch_size, num_chunks, self.chunk_size, d_model).
                - chunked_mask (batch_size, num_chunks, 1, self.chunk_size) or None.
                - padding_needed (int): Amount of padding added to the sequence length.
                - original_seq_len (int): Sequence length before padding.
        """
        batch_size, original_seq_len, d_model = input_tensor.shape
        
        padding_needed = 0
        if original_seq_len == 0: # Handle empty sequences
             # If sequence is empty, num_chunks will be 0. Avoid division by zero.
             # Return empty chunks or raise error, depending on desired behavior.
             # For now, let it proceed, num_chunks will be 0.
             pass
        elif original_seq_len % self.chunk_size != 0:
            padding_needed = self.chunk_size - (original_seq_len % self.chunk_size)
        
        if padding_needed > 0:
            # Pad input_tensor: (B, S, D) -> (B, S_padded, D)
            input_tensor = F.pad(input_tensor, (0, 0, 0, padding_needed)) 
            if input_tensor_mask is not None:
                # Pad input_tensor_mask: (B, 1, 1, S) -> (B, 1, 1, S_padded)
                input_tensor_mask = F.pad(input_tensor_mask, (0, padding_needed), mode='constant', value=0.0)

        padded_seq_len = input_tensor.shape[1]
        if self.chunk_size == 0 : # Should be caught by __init__ validation
            raise ValueError("chunk_size cannot be zero in _chunk_input")
        num_chunks = padded_seq_len // self.chunk_size

        chunked_tensor = input_tensor.view(batch_size, num_chunks, self.chunk_size, d_model)
        
        chunked_mask_output = None
        if input_tensor_mask is not None:
            # Reshape mask (B, 1, 1, S_padded) to (B, N, 1, C) for local MHA
            # S_padded = N * C
            # (B, 1, 1, N*C) -> (B, 1, N, C) -> (B, N, 1, C)
            chunked_mask_output = input_tensor_mask.view(batch_size, 1, num_chunks, self.chunk_size) 
            chunked_mask_output = chunked_mask_output.permute(0, 2, 1, 3) 
            
        return chunked_tensor, chunked_mask_output, padding_needed, original_seq_len

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, input_padding_mask: torch.Tensor = None):
        batch_size, original_seq_len, d_model = query.shape
        device = query.device

        if self.chunk_size <= 0: 
            raise ValueError("AHCAttention.chunk_size must be positive (runtime check).")
        # Runtime check if original_seq_len is 0 could be useful if not handled by _chunk_input robustly.
        # if original_seq_len == 0:
        #     # Return query or appropriately shaped zeros, and padding_info
        #     return query, {"padding_added_to_input": 0, "original_input_seq_len": 0}


        # 0. Prepare input_padding_mask for chunking
        processed_input_mask_for_chunking = AHCAttention._process_input_mask(
            mask=input_padding_mask,
            target_seq_len=original_seq_len,
            device=device,
            target_dtype=torch.float
        )

        # 1. Chunking Input Tensor and Mask (assuming Q, K, V are same for self-attention)
        chunked_query, chunked_padding_mask_for_local_mha, padding_needed, _ = \
            self._chunk_input(query, processed_input_mask_for_chunking)
        
        # Handle case where original_seq_len is less than chunk_size, resulting in num_chunks=0 if not padded first
        # _chunk_input now handles padding to at least one chunk if original_seq_len > 0.
        # If original_seq_len was 0, chunked_query might be (B,0,C,D).
        if chunked_query.shape[1] == 0 and original_seq_len > 0 : # Should not happen if _chunk_input pads correctly
             raise RuntimeError("_chunk_input failed to produce chunks for non-empty sequence.")
        elif original_seq_len == 0: # If input was empty, return empty.
            return query, {"padding_added_to_input": padding_needed, "original_input_seq_len": original_seq_len}


        current_num_chunks = chunked_query.shape[1]

        # 2. Local Attention within Chunks
        local_mha_input = chunked_query.contiguous().view(-1, self.chunk_size, d_model)
        sz_c = self.chunk_size
        causal_mask_for_local_attn = torch.tril(torch.ones(sz_c, sz_c, device=device, dtype=torch.float)).unsqueeze(0).unsqueeze(0)
        
        final_local_mask = causal_mask_for_local_attn
        if chunked_padding_mask_for_local_mha is not None:
            # chunked_padding_mask_for_local_mha is (B, N, 1, C)
            # Reshape to (B*N, 1, 1, C) for broadcasting with causal_mask_for_local_attn (1,1,C,C)
            prepared_padding_mask = chunked_padding_mask_for_local_mha.contiguous().view(
                batch_size * current_num_chunks, 1, 1, sz_c
            )
            final_local_mask = causal_mask_for_local_attn * prepared_padding_mask # Element-wise product
        
        local_context_flat, _ = self.local_mha(
            local_mha_input, local_mha_input, local_mha_input, mask=final_local_mask
        )
        local_context = local_context_flat.view(batch_size, current_num_chunks, self.chunk_size, d_model)

        # 3. Summarization
        summary_tokens = None
        summarization_type = self.summarization_method_config.get('type')
        if summarization_type == "S1_pool":
            summary_tokens = self._summarize_s1_pool(local_context, self.summarization_method_config)
        elif summarization_type == "S2_linear_on_pool":
            summary_tokens = self._summarize_s2_linear_on_pool(local_context, self.summarization_method_config)
        else: # Should be caught by __init__ validation
            raise ValueError(f"Unsupported summarization type configured: {summarization_type}")

        # 4. Global Attention
        global_context_summaries = None
        global_attention_type = self.global_attention_method_config.get('type')

        # Create summary_validity_mask (P0 LlamaPReview fix)
        summary_validity_mask = None # (B, N_chunks)
        if chunked_padding_mask_for_local_mha is not None:
            # A summary is valid if its corresponding chunk had at least one non-padded token.
            # chunked_padding_mask_for_local_mha is (B, N, 1, C), 1 for keep, 0 for pad.
            squeezed_chunk_padding_mask = chunked_padding_mask_for_local_mha.squeeze(2) # (B, N, C)
            summary_validity_mask = torch.any(squeezed_chunk_padding_mask == 1.0, dim=-1).float() # (B, N)
        else:
            # If no input padding mask, all chunks and their summaries are considered valid.
            summary_validity_mask = torch.ones(batch_size, current_num_chunks, device=device, dtype=torch.float)

        if global_attention_type == "G1_full_mha":
            if self.global_mha is None: # Should be caught by __init__
                raise RuntimeError("G1_full_mha configured, but global_mha module is not initialized.")
            if summary_tokens is None: # Should be caught if summarization_type was invalid
                raise ValueError("Summary tokens are None, cannot apply global attention.")
            
            num_summary_tokens = summary_tokens.shape[1]
            global_causal_mask = torch.tril(torch.ones(num_summary_tokens, num_summary_tokens, device=device, dtype=torch.float)).unsqueeze(0).unsqueeze(0)
            
            # Prepare summary_validity_mask for MHA: (B, N) -> (B, 1, 1, N_keys=N)
            # This will be multiplied with the (1,1,N_queries,N_keys) causal mask.
            # Where summary_validity_mask is 0, the corresponding key summary token will be masked out.
            prepared_summary_validity_mask = summary_validity_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
            
            final_global_mask = global_causal_mask * prepared_summary_validity_mask
            
            global_context_summaries, _ = self.global_mha(summary_tokens, summary_tokens, summary_tokens, mask=final_global_mask)
        elif global_attention_type is None: # Pass-through
            if summary_tokens is None:
                 raise ValueError("Summary tokens are None, and no global attention is configured.")
            global_context_summaries = summary_tokens
        # Other global_attention_types would have been caught by __init__ or need specific module here

        # 5. Combination
        if local_context is None or global_context_summaries is None:
            raise ValueError("Local context or global context summaries are None, cannot apply combination strategy.")
        
        combined_context_chunked = self.combination_strategy.apply(local_context, global_context_summaries, self.chunk_size)

        # 6. Reshape to final output dimensions and unpad
        output_tensor = combined_context_chunked.contiguous().view(batch_size, -1, d_model)
        
        if padding_needed > 0 and original_seq_len > 0 : # only slice if there was an original sequence
            output_tensor = output_tensor[:, :original_seq_len, :]
        elif original_seq_len == 0: # if original was empty, output should match that (empty)
            output_tensor = torch.empty((batch_size, 0, d_model), device=device, dtype=query.dtype)


        padding_info = {
            "padding_added_to_input": padding_needed,
            "original_input_seq_len": original_seq_len
        }
        return output_tensor, padding_info