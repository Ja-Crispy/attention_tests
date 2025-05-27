import torch
import torch.nn as nn
import torch.nn.functional as F # Added import for F.pad
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

        processed_input_mask_for_chunking = None
        if input_padding_mask is not None:
            if input_padding_mask.ndim == 2:
                processed_input_mask_for_chunking = input_padding_mask.float().unsqueeze(1).unsqueeze(2)
            elif input_padding_mask.ndim == 4:
                processed_input_mask_for_chunking = input_padding_mask.float()
            else:
                raise ValueError(f"Unsupported input_padding_mask shape: {input_padding_mask.shape}")

        chunked_query, chunked_padding_mask_for_local_mha, padding_needed, _ = \
            self._chunk_input(query, processed_input_mask_for_chunking)
        current_num_chunks = chunked_query.shape[1]

        local_mha_input = chunked_query.contiguous().view(-1, self.chunk_size, d_model)
        
        sz_c = self.chunk_size
        causal_mask_for_local_attn = torch.tril(torch.ones(sz_c, sz_c, device=device, dtype=torch.float)).unsqueeze(0).unsqueeze(0)

        final_local_mask = causal_mask_for_local_attn 
        if chunked_padding_mask_for_local_mha is not None:
            prepared_padding_mask = chunked_padding_mask_for_local_mha.contiguous().view(
                batch_size * current_num_chunks, 1, 1, sz_c
            )
            final_local_mask = causal_mask_for_local_attn * prepared_padding_mask
            
        local_context_flat, _ = self.local_mha(
            local_mha_input, local_mha_input, local_mha_input, mask=final_local_mask
        )
        local_context = local_context_flat.view(batch_size, current_num_chunks, self.chunk_size, d_model)

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

        global_context_summaries = None
        global_attention_type = self.global_attention_method_config.get('type')

        if self.global_mha and global_attention_type == "G1_full_mha":
            if summary_tokens is None: 
                raise ValueError("Summary tokens are None, cannot apply G1_full_mha global attention.")
            
            num_summary_tokens = summary_tokens.shape[1] 
            global_causal_mask = torch.tril(torch.ones(num_summary_tokens, num_summary_tokens, device=device)).unsqueeze(0).unsqueeze(0)
            global_context_summaries, _ = self.global_mha(summary_tokens, summary_tokens, summary_tokens, mask=global_causal_mask)
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

        output_tensor = combined_context_chunked.contiguous().view(batch_size, -1, d_model)
        
        if padding_needed > 0:
            output_tensor = output_tensor[:, :original_seq_len, :]
        
        return output_tensor, None
