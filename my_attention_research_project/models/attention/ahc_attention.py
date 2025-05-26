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

        # Store any additional kwargs if needed, or process them
        # For example, these might contain specific parameters for summarization or global attention
        # that are not explicitly listed in the signature but are passed via attention_config.
        self.additional_config = kwargs

        # Instantiate Local MHA
        self.local_mha = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )

        # Initialize Global MHA
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
        elif global_attention_type is not None: # If a type is specified but not G1
            raise NotImplementedError(f"Global attention type '{global_attention_type}' is not yet implemented.")

        # Initialize Summarization-specific layers (e.g., for S2)
        self.s2_linear_projection = None
        if self.summarization_method_config.get('type') == "S2_linear_on_pool":
            if not hasattr(self, '_summarize_s1_pool'): # Check if S1 is available for reuse
                raise AttributeError("S2_linear_on_pool requires S1_pool's pooling logic. Ensure _summarize_s1_pool is defined.")
            self.s2_linear_projection = nn.Linear(self.d_model, self.d_model)
        
        # Placeholder for other module instantiations (combination)
        # e.g., Initialize combination layer based on combination_method_config

    def _combine_c1_broadcast_add(self, local_context: torch.Tensor, global_context_summaries: torch.Tensor) -> torch.Tensor:
        """
        Combines local and global contexts using broadcast addition (C1).
        Args:
            local_context (torch.Tensor): Output of local attention.
                                          Shape: (batch_size, num_chunks, chunk_size, d_model).
            global_context_summaries (torch.Tensor): Processed summary tokens from global attention.
                                                     Shape: (batch_size, num_chunks, d_model).
        Returns:
            torch.Tensor: Combined context. Shape: (batch_size, num_chunks, chunk_size, d_model).
        """
        # global_context_summaries is (B, N, D)
        # We need to expand it to (B, N, C, D) to match local_context (B, N, C, D)
        # C is self.chunk_size
        expanded_global_context = global_context_summaries.unsqueeze(2).expand(-1, -1, self.chunk_size, -1)
        combined_context = local_context + expanded_global_context
        return combined_context

    def _summarize_s1_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        """
        Performs pooling-based summarization (S1).
        Args:
            local_context (torch.Tensor): Output of local attention.
                                          Shape: (batch_size, num_chunks, chunk_size, d_model).
            summarization_config (dict): Configuration for summarization, containing 'pool_type'.

        Returns:
            torch.Tensor: Summary tokens. Shape: (batch_size, num_chunks, d_model).
        """
        pool_type = summarization_config.get('pool_type', 'mean') # Default to mean if not specified

        if pool_type == 'mean':
            summary_tokens = torch.mean(local_context, dim=2) # Pool over chunk_size dimension
        elif pool_type == 'max':
            summary_tokens = torch.max(local_context, dim=2).values # Pool over chunk_size dimension
        else:
            raise ValueError(f"Unsupported S1 pool_type: {pool_type}")
        
        return summary_tokens

    def _summarize_s2_linear_on_pool(self, local_context: torch.Tensor, summarization_config: dict) -> torch.Tensor:
        """
        Performs S2 summarization (Linear projection on pooled tokens).
        Args:
            local_context (torch.Tensor): Output of local attention.
                                          Shape: (batch_size, num_chunks, chunk_size, d_model).
            summarization_config (dict): Configuration for summarization.
                                         'pool_type' from this config is used by _summarize_s1_pool.
        Returns:
            torch.Tensor: Summary tokens. Shape: (batch_size, num_chunks, d_model).
        """
        # Reuse S1's pooling logic by calling it directly
        pooled_tokens = self._summarize_s1_pool(local_context, summarization_config)
        # pooled_tokens shape: (batch_size, num_chunks, d_model)

        if self.s2_linear_projection is None:
            raise RuntimeError("S2 linear projection layer is not initialized. Check configuration.")
            
        summary_tokens = self.s2_linear_projection(pooled_tokens)
        return summary_tokens

    def _chunk_input(self, x: torch.Tensor):
        """
        Pads and reshapes the input tensor into chunks.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tuple[torch.Tensor, int, int]:
                - chunked_x (torch.Tensor): Reshaped tensor (batch_size, num_chunks, chunk_size, d_model).
                - padding_needed (int): Amount of padding added.
                - original_seq_len (int): Sequence length before padding.
        """
        batch_size, seq_len, d_model = x.shape
        original_seq_len = seq_len

        padding_needed = 0
        if seq_len % self.chunk_size != 0:
            padding_needed = self.chunk_size - (seq_len % self.chunk_size)
            # Pad with zeros on the right (sequence dimension)
            # F.pad format: (pad_left_dimN, pad_right_dimN, pad_left_dimN-1, pad_right_dimN-1, ...)
            # For (B, S, D), we want to pad S (dim 1).
            # So, (0,0) for d_model (last dim), and (0, padding_needed) for seq_len (2nd to last dim).
            x = F.pad(x, (0, 0, 0, padding_needed))
        
        padded_seq_len = x.shape[1] # seq_len after padding
        num_chunks = padded_seq_len // self.chunk_size

        chunked_x = x.view(batch_size, num_chunks, self.chunk_size, d_model)
        return chunked_x, padding_needed, original_seq_len

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for AHCAttention.
        Args:
            query (torch.Tensor): Query tensor. Shape: (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor. Shape: (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor. Shape: (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask. Its shape depends on the specific
                                           attention mechanism and how it's applied (e.g., for
                                           local MHA or global attention). Defaults to None.
                                           Currently unused in this placeholder implementation.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context_vector (torch.Tensor): The output of the attention mechanism.
                                                 Shape: (batch_size, seq_len, d_model).
                - attention_weights (torch.Tensor, optional): Attention weights. Shape varies.
                                                              Can be None.
        """
        # Assuming query, key, value are the same for self-attention for now.
        # This will change as AHC logic is built out.
        # For now, operate on `query` as the primary input `x`.
        # In a full Transformer, query, key, value might be different for cross-attention.
        # For encoder self-attention, they are typically the same.
        x = query
        x = query # Assuming Q, K, V are same for self-attention
        batch_size, original_seq_len, d_model = x.shape
        device = x.device

        # 1. Chunking
        chunked_x, padding_needed, _ = self._chunk_input(x)
        # chunked_x shape: (B, N, C, D) where N=num_chunks, C=chunk_size
        # Store num_chunks for potential use, e.g. if not inferable from a tensor later
        current_num_chunks = chunked_x.shape[1]


        # 2. Local Attention
        local_mha_input = chunked_x.contiguous().view(-1, self.chunk_size, d_model)
        # Causal mask for local attention
        sz_c = self.chunk_size
        causal_mask_for_local_attn = torch.tril(torch.ones(sz_c, sz_c, device=device)).unsqueeze(0).unsqueeze(0)
        
        local_context_flat, _ = self.local_mha(local_mha_input, local_mha_input, local_mha_input, mask=causal_mask_for_local_attn)
        local_context = local_context_flat.view(batch_size, current_num_chunks, self.chunk_size, d_model)
        # local_context shape: (B, N, C, D)

        # 3. Summarization
        summary_tokens = None
        summarization_type = self.summarization_method_config.get('type')
        if summarization_type == "S1_pool":
            summary_tokens = self._summarize_s1_pool(local_context, self.summarization_method_config)
        elif summarization_type == "S2_linear_on_pool":
            summary_tokens = self._summarize_s2_linear_on_pool(local_context, self.summarization_method_config)
        elif summarization_type is None: # Explicitly check for None
             raise ValueError("Summarization type not specified in summarization_method_config.")
        else: # Handles any other unsupported type
            raise ValueError(f"Unsupported summarization type: {summarization_type}")
        # summary_tokens shape: (B, N, D)

        # 4. Global Attention
        global_context_summaries = None
        global_attention_type = self.global_attention_method_config.get('type')

        if self.global_mha and global_attention_type == "G1_full_mha":
            if summary_tokens is None: 
                raise ValueError("Summary tokens are None, cannot apply G1_full_mha global attention.")
            
            num_summary_tokens = summary_tokens.shape[1] # Should be current_num_chunks
            global_causal_mask = torch.tril(torch.ones(num_summary_tokens, num_summary_tokens, device=device)).unsqueeze(0).unsqueeze(0)
            global_context_summaries, _ = self.global_mha(summary_tokens, summary_tokens, summary_tokens, mask=global_causal_mask)
        elif global_attention_type is None: # No global attention configured
            if summary_tokens is None:
                raise ValueError("Summary tokens are None, and no global attention is configured to pass them through.")
            global_context_summaries = summary_tokens # Pass through
        elif global_attention_type == "G1_full_mha" and not self.global_mha:
             raise ValueError("G1_full_mha global attention configured but module not initialized.")
        else: # Other types or misconfigurations
            if summary_tokens is None:
                 raise ValueError(f"Global attention type {global_attention_type} requires summary tokens, but they are None.")
            # For other specified but not implemented global attention types, we might pass-through or error
            # Current __init__ errors for other types, so this path implies pass-through for safety if reached.
            print(f"Warning: Global attention type '{global_attention_type}' logic not fully implemented or not G1. Passing summary_tokens.")
            global_context_summaries = summary_tokens
        # global_context_summaries shape: (B, N, D)

        # 5. Combination
        combined_context_chunked = None
        combination_type = self.combination_method_config.get('type')
        if combination_type == "C1_broadcast_add":
            if global_context_summaries is None:
                 raise ValueError("Global context summaries are None, cannot apply C1_broadcast_add combination.")
            if local_context is None: # Should not happen given the flow
                 raise ValueError("Local context is None, cannot apply C1_broadcast_add combination.")
            combined_context_chunked = self._combine_c1_broadcast_add(local_context, global_context_summaries)
        elif combination_type is None:
            raise ValueError("Combination type not specified in combination_method_config.")
        else:
            raise ValueError(f"Unsupported combination type: {combination_type}")
        # combined_context_chunked shape: (B, N, C, D)

        # 6. Reshape to final output dimensions
        # Reshape from (B, N, C, D) to (B, N*C, D)
        output_tensor = combined_context_chunked.contiguous().view(batch_size, -1, d_model)
        
        # Remove padding
        if padding_needed > 0:
            output_tensor = output_tensor[:, :original_seq_len, :]
        
        return output_tensor, None # Final output, None for attention_weights
        # This mask is (1, 1, chunk_size, chunk_size) and will be broadcasted by MHA.
        # `torch.tril` creates a lower triangular matrix (1s in lower triangle, 0s in upper).
        # `ScaledDotProductAttention` in `vanilla_mha` uses `mask == 0` to mask positions.
        # So, this mask correctly allows attention to previous and current positions within a chunk.
        sz = self.chunk_size
        causal_mask_for_local_attn = torch.tril(torch.ones(sz, sz, device=x.device)).unsqueeze(0).unsqueeze(0)
        # Shape of causal_mask_for_local_attn: (1, 1, self.chunk_size, self.chunk_size)

        # Apply local MHA
        # local_context shape: (batch_size * num_chunks, self.chunk_size, d_model)
        # We pass local_mha_input for Q, K, V as it's self-attention within chunks.
        local_context, _ = self.local_mha(
            query=local_mha_input, 
            key=local_mha_input, 
            value=local_mha_input, 
            mask=causal_mask_for_local_attn
        )
        
