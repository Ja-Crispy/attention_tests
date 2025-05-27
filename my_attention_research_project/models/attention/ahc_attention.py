import torch
import torch.nn as nn
import torch.nn.functional as F # Added import for F.pad
import warnings # Added for chunk_size validation warning
from .vanilla_mha import MultiHeadAttention # Used for local and potentially global attention

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
        # Optional: if model_max_length is passed and validated here (e.g. from TransformerEncoder)
        # model_max_length = kwargs.get('model_max_length')
        # if model_max_length and self.chunk_size > model_max_length:
        #     raise ValueError(f"AHCAttention chunk_size ({self.chunk_size}) > model_max_length ({model_max_length})")

        self.local_mha = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )

        self.global_mha = None
        global_attention_type = self.global_attention_method_config.get('type')
        if global_attention_type == "G1_full_mha":
            global_n_heads = self.global_attention_method_config.get('num_heads', self.n_heads)
            global_dropout_rate = self.global_attention_method_config.get('dropout_rate', self.dropout_rate)
            self.global_mha = MultiHeadAttention(
                d_model=self.d_model,
                n_heads=global_n_heads,
                dropout_rate=global_dropout_rate
            )
        elif global_attention_type is not None:
            raise NotImplementedError(f"Global attention type '{global_attention_type}' is not yet implemented.")

        self.s2_linear_projection = None
        if self.summarization_method_config.get('type') == "S2_linear_on_pool":
            if not hasattr(self, '_summarize_s1_pool'):
                raise AttributeError("S2_linear_on_pool requires S1_pool's pooling logic. Ensure _summarize_s1_pool is defined.")
            self.s2_linear_projection = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def _process_input_mask(mask: torch.Tensor, target_seq_len: int, device: torch.device, target_dtype: torch.dtype = torch.float) -> torch.Tensor:
        """
        Processes an input attention mask into a canonical format suitable for chunking.

        This utility converts common padding mask formats (2D or a specific 4D)
        into a standard 4D tensor shape (batch_size, 1, 1, seq_len) with a float dtype,
        where 1.0 means keep and 0.0 means pad. This format is expected by the 
        _chunk_input method for masks.

        Args:
            mask (torch.Tensor, optional): The input attention mask. Supported shapes:
                - None: Returns None.
                - 2D (batch_size, seq_len): Boolean or float/int. True/1 indicates keep.
                - 4D (batch_size, 1, 1, seq_len): Float. 1.0 indicates keep.
            target_seq_len (int): The expected sequence length for the output mask,
                                  typically from the main input tensor (e.g., query).
            device (torch.device): The target device for the output mask.
            target_dtype (torch.dtype, optional): The target data type for the output mask.
                                                 Defaults to torch.float.

        Returns:
            torch.Tensor, optional: The processed mask in shape (batch_size, 1, 1, seq_len)
                                    and target_dtype, or None if the input mask was None.

        Raises:
            ValueError: If the input mask has an unsupported dimension or its sequence
                        length (if 2D) does not match target_seq_len.
                        Also raises ValueError for 4D masks not in the expected
                        (B, 1, 1, S) format, as this function is specialized for
                        creating chunkable padding masks, not for general 4D MHA mask transformations.
        """
        if mask is None:
            return None

        # Ensure mask is float (0.0 for pad, 1.0 for keep) before shape manipulation
        if mask.dtype == torch.bool:
            mask = mask.type(target_dtype)  # True -> 1.0, False -> 0.0
        elif mask.dtype != target_dtype: # If already float but not target_dtype (e.g. float64)
            mask = mask.type(target_dtype)

        if mask.ndim == 2:
            # Input: (batch_size, seq_len)
            if mask.shape[1] != target_seq_len:
                raise ValueError(
                    f"Input 2D mask's sequence length ({mask.shape[1]}) does not match "
                    f"target sequence length ({target_seq_len})."
                )
            # Reshape to (batch_size, 1, 1, seq_len)
            processed_mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.ndim == 4:
            # Expected input: (batch_size, 1, 1, seq_len)
            if mask.shape[1] == 1 and mask.shape[2] == 1 and mask.shape[3] == target_seq_len:
                processed_mask = mask
            else:
                # This utility is specifically for preparing a simple padding mask for chunking.
                # It does not handle more complex 4D MHA masks (e.g., B,H,Q,K or B,1,Q,K where Q!=1).
                raise ValueError(
                    f"Input 4D mask has shape {mask.shape}. Expected (batch_size, 1, 1, {target_seq_len}) "
                    f"for this utility. Other 4D mask formats require custom handling."
                )
        else:
            raise ValueError(
                f"Unsupported mask dimension: {mask.ndim}. Expected 2D (batch_size, seq_len) "
                f"or 4D (batch_size, 1, 1, seq_len)."
            )

        # Ensure correct device
        if processed_mask.device != device:
            processed_mask = processed_mask.to(device)
            
        return processed_mask

    def _combine_c1_broadcast_add(self, local_context: torch.Tensor, global_context_summaries: torch.Tensor) -> torch.Tensor:
        expanded_global_context = global_context_summaries.unsqueeze(2).expand(-1, -1, self.chunk_size, -1)
        combined_context = local_context + expanded_global_context
        return combined_context

    def _summarize_s1_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        pool_type = summarization_config.get('pool_type', 'mean')
        if pool_type == 'mean':
            summary_tokens = torch.mean(local_context, dim=2)
        elif pool_type == 'max':
            summary_tokens = torch.max(local_context, dim=2).values
        else:
            raise ValueError(f"Unsupported S1 pool_type: {pool_type}")
        return summary_tokens

    def _summarize_s2_linear_on_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        pooled_tokens = self._summarize_s1_pool(local_context, summarization_config)
        if self.s2_linear_projection is None:
            raise RuntimeError("S2 linear projection layer is not initialized. Check configuration.")
        summary_tokens = self.s2_linear_projection(pooled_tokens)
        return summary_tokens

    def _chunk_input(self, input_tensor: torch.Tensor, input_tensor_mask: torch.Tensor = None):
        """
        Pads and reshapes the input tensor and its mask into chunks.
        Pads the input tensor and mask along the sequence length dimension 
        if original_seq_len is not a multiple of chunk_size, ensuring all chunks are full.

        Args:
            input_tensor (torch.Tensor): Input tensor.
                                         Shape: (batch_size, original_seq_len, d_model).
            input_tensor_mask (torch.Tensor, optional): Mask for input_tensor. 
                                                        Expected shape: (batch_size, 1, 1, original_seq_len). 
                                                        Values: 0.0 for pad/mask, 1.0 for keep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None, int, int]:
                - chunked_tensor (torch.Tensor): Reshaped tensor.
                                                 Shape: (batch_size, num_chunks, self.chunk_size, d_model).
                - chunked_mask (torch.Tensor | None): Reshaped mask or None.
                                                      Shape: (batch_size, num_chunks, 1, self.chunk_size).
                - padding_needed (int): Amount of padding added to the sequence length.
                - original_seq_len (int): Sequence length before padding.
        """
        batch_size, original_seq_len, d_model = input_tensor.shape
        
        padding_needed = 0
        if original_seq_len % self.chunk_size != 0:
            padding_needed = self.chunk_size - (original_seq_len % self.chunk_size)
        
        # Apply padding if needed
        if padding_needed > 0:
            input_tensor = F.pad(input_tensor, (0, 0, 0, padding_needed)) 
            if input_tensor_mask is not None:
                input_tensor_mask = F.pad(input_tensor_mask, (0, padding_needed), mode='constant', value=0.0)

        padded_seq_len = input_tensor.shape[1]
        num_chunks = padded_seq_len // self.chunk_size

        chunked_tensor = input_tensor.view(batch_size, num_chunks, self.chunk_size, d_model)
        
        chunked_mask_output = None
        if input_tensor_mask is not None:
            chunked_mask_output = input_tensor_mask.view(batch_size, 1, num_chunks, self.chunk_size) 
            chunked_mask_output = chunked_mask_output.permute(0, 2, 1, 3) 
            
        return chunked_tensor, chunked_mask_output, padding_needed, original_seq_len

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, input_padding_mask: torch.Tensor = None):
        batch_size, original_seq_len, d_model = query.shape
        device = query.device

        # Runtime check for chunk_size vs input sequence length
        if self.chunk_size <= 0: # Should have been caught in __init__, but for safety.
            raise ValueError("AHCAttention.chunk_size must be positive (runtime check).")
        if self.chunk_size > original_seq_len:
            warnings.warn(
                f"AHCAttention: chunk_size ({self.chunk_size}) is greater than input sequence length ({original_seq_len}). "
                f"Input will be processed as a single chunk with padding.",
                UserWarning
            )

        # 0. Prepare input_padding_mask (if provided)
        # Input input_padding_mask shape: (B, S_orig) boolean/float or (B, 1, 1, S_orig) float
        processed_input_mask_for_chunking = AHCAttention._process_input_mask(
            mask=input_padding_mask,
            target_seq_len=original_seq_len,
            device=device,
            target_dtype=torch.float # MHA expects float mask (0 for mask, 1 for keep)
        )
        # Output processed_input_mask_for_chunking shape: (B, 1, 1, S_orig) float or None

        # 1. Chunking Input Tensor and Mask
        # Input query shape: (batch_size, original_seq_len, d_model)
        chunked_query, chunked_padding_mask_for_local_mha, padding_needed, _ = \
            self._chunk_input(query, processed_input_mask_for_chunking)
        # Output chunked_query shape: (batch_size, current_num_chunks, self.chunk_size, self.d_model)
        # Output chunked_padding_mask_for_local_mha shape: (batch_size, current_num_chunks, 1, self.chunk_size) or None
        current_num_chunks = chunked_query.shape[1]

        # 2. Local Attention within Chunks
        # Input local_mha_input shape: (batch_size * current_num_chunks, self.chunk_size, self.d_model)
        local_mha_input = chunked_query.contiguous().view(-1, self.chunk_size, d_model)
        
        sz_c = self.chunk_size
        # Base causal mask for local attention (1, 1, C, C), 1s for keep, 0s for mask
        causal_mask_for_local_attn = torch.tril(torch.ones(sz_c, sz_c, device=device, dtype=torch.float)).unsqueeze(0).unsqueeze(0)

        final_local_mask = causal_mask_for_local_attn 
        if chunked_padding_mask_for_local_mha is not None:
            # chunked_padding_mask_for_local_mha is (B, N, 1, C). Values are 0.0 for pad, 1.0 for keep.
            # Reshape to (B*N, 1, 1, C) for MHA.
            prepared_padding_mask = chunked_padding_mask_for_local_mha.contiguous().view(
                batch_size * current_num_chunks, 1, 1, sz_c
            )
            # Combine: if either mask is 0 (mask), the result is 0 (mask).
            final_local_mask = causal_mask_for_local_attn * prepared_padding_mask
        # Input final_local_mask shape: (B*N, 1, 1, C) or (1, 1, C, C)
            
        local_context_flat, _ = self.local_mha(
            local_mha_input, local_mha_input, local_mha_input, mask=final_local_mask
        )
        # Output local_context_flat shape: (batch_size * current_num_chunks, self.chunk_size, self.d_model)
        local_context = local_context_flat.view(batch_size, current_num_chunks, self.chunk_size, d_model)
        # Output local_context shape: (batch_size, current_num_chunks, self.chunk_size, self.d_model)

        # 3. Summarization
        # Input local_context shape: (batch_size, current_num_chunks, self.chunk_size, self.d_model)
        summary_tokens = None
        summarization_type = self.summarization_method_config.get('type')
        if summarization_type == "S1_pool":
            summary_tokens = self._summarize_s1_pool(local_context, self.summarization_method_config)
        elif summarization_type == "S2_linear_on_pool":
            summary_tokens = self._summarize_s2_linear_on_pool(local_context, self.summarization_method_config)
        elif summarization_type is None: 
             raise ValueError("Summarization type not specified in summarization_method_config.")
        else: 
            raise ValueError(f"Unsupported summarization type: {summarization_type}")
        # Output summary_tokens shape: (batch_size, current_num_chunks, self.d_model)

        # 4. Global Attention
        # Input summary_tokens shape: (batch_size, current_num_chunks, self.d_model)
        summary_validity_mask = None
        if chunked_padding_mask_for_local_mha is not None:
            squeezed_chunk_padding_mask = chunked_padding_mask_for_local_mha.squeeze(2) # (B, N, C)
            summary_validity_mask = torch.any(squeezed_chunk_padding_mask != 0.0, dim=-1).float() # (B, N)
        else:
            summary_validity_mask = torch.ones(batch_size, current_num_chunks, device=device, dtype=torch.float)
        # summary_validity_mask shape: (batch_size, current_num_chunks)
        
        global_context_summaries = None
        global_attention_type = self.global_attention_method_config.get('type')

        if self.global_mha and global_attention_type == "G1_full_mha":
            if summary_tokens is None:
                raise ValueError("Summary tokens are None, cannot apply global attention.")
            
            num_summary_tokens = summary_tokens.shape[1] 

            global_causal_mask = torch.tril(torch.ones(num_summary_tokens, num_summary_tokens, device=device, dtype=torch.float)).unsqueeze(0).unsqueeze(0)
            prepared_summary_validity_mask = summary_validity_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
            final_global_mask = global_causal_mask * prepared_summary_validity_mask 
            # final_global_mask shape: (B, 1, N, N) or (1, 1, N, N) * (B, 1, 1, N) -> (B, 1, N, N) (broadcasted)

            global_context_summaries, _ = self.global_mha(summary_tokens, summary_tokens, summary_tokens, mask=final_global_mask)
        elif global_attention_type is None: 
            if summary_tokens is None:
                 raise ValueError("Summary tokens are None, and no global attention is configured to pass them through.")
            global_context_summaries = summary_tokens
        elif global_attention_type == "G1_full_mha" and not self.global_mha: 
             raise ValueError("G1_full_mha global attention configured but module not initialized.")
        else: 
            if summary_tokens is None:
                 raise ValueError(f"Global attention type {global_attention_type} requires summary tokens, but they are None.")
            print(f"Warning: Global attention type '{global_attention_type}' logic not fully implemented or not G1. Passing summary_tokens.")
            global_context_summaries = summary_tokens
        # Output global_context_summaries shape: (batch_size, current_num_chunks, self.d_model)
        
        # 5. Combination
        # Input local_context shape: (batch_size, current_num_chunks, self.chunk_size, self.d_model)
        # Input global_context_summaries shape: (batch_size, current_num_chunks, self.d_model)
        combined_context_chunked = None
        combination_type = self.combination_method_config.get('type')
        if combination_type == "C1_broadcast_add":
            if global_context_summaries is None:
                 raise ValueError("Global context summaries are None, cannot apply C1_broadcast_add combination.")
            if local_context is None: 
                 raise ValueError("Local context is None, cannot apply C1_broadcast_add combination.")
            combined_context_chunked = self._combine_c1_broadcast_add(local_context, global_context_summaries)
        elif combination_type is None:
            raise ValueError("Combination type not specified in combination_method_config.")
        else:
            raise ValueError(f"Unsupported combination type: {combination_type}")
        # Output combined_context_chunked shape: (batch_size, current_num_chunks, self.chunk_size, self.d_model)

        # 6. Reshape to final output dimensions
        output_tensor = combined_context_chunked.contiguous().view(batch_size, -1, d_model)
        # Output output_tensor (before unpadding) shape: (batch_size, current_num_chunks * self.chunk_size, self.d_model)
        
        if padding_needed > 0:
            output_tensor = output_tensor[:, :original_seq_len, :]
        # Final output_tensor shape: (batch_size, original_seq_len, self.d_model)
        
        return output_tensor, None
