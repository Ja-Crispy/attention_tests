import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention.

    Args:
        d_k (int): Dimension of the key and query vectors.
    """
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Scaled Dot-Product Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_v).
                                  seq_len_k and seq_len_v must be the same.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, 1, seq_len_k) or
                                           (batch_size, 1, seq_len_q, seq_len_k) for broadcasting.
                                           Defaults to None.

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, seq_len_q, d_v).
            attention_weights (torch.Tensor): Attention weights tensor of shape (batch_size, seq_len_q, seq_len_k).
        """
        # MatMul QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # MatMul attention_weights * V
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_model (int): Total dimension of the model.
        n_heads (int): Number of attention heads.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
    """
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model) # d_model -> n_heads * d_k
        self.W_k = nn.Linear(d_model, d_model) # d_model -> n_heads * d_k
        self.W_v = nn.Linear(d_model, d_model) # d_model -> n_heads * d_v

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(self.d_k)

        # Output linear layer
        self.W_o = nn.Linear(d_model, d_model) # n_heads * d_v -> d_model

        # Dropout layer
        self.dropout = None
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
                                 seq_len_k and seq_len_v must be the same.
            mask (torch.Tensor, optional): Mask tensor. Its shape depends on the attention mechanism
                                           (e.g., padding mask, causal mask).
                                           For self-attention with padding, shape could be (batch_size, 1, 1, seq_len_k).
                                           For causal self-attention, shape could be (1, 1, seq_len_q, seq_len_k).
                                           Defaults to None.

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, seq_len_q, d_model).
            attention_weights (torch.Tensor): Attention weights tensor of shape (batch_size, n_heads, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)

        # 1. Linear projections
        q_proj = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        k_proj = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        v_proj = self.W_v(value)  # (batch_size, seq_len_v, d_model)

        # 2. Reshape for multi-head: (batch_size, seq_len, n_heads, d_k/d_v)
        q_reshaped = q_proj.view(batch_size, -1, self.n_heads, self.d_k)
        k_reshaped = k_proj.view(batch_size, -1, self.n_heads, self.d_k)
        v_reshaped = v_proj.view(batch_size, -1, self.n_heads, self.d_v)

        # 3. Transpose to: (batch_size, n_heads, seq_len, d_k/d_v)
        q_transposed = q_reshaped.transpose(1, 2)
        k_transposed = k_reshaped.transpose(1, 2)
        v_transposed = v_reshaped.transpose(1, 2)

        # Adjust mask for multi-head attention if it's not already prepared
        if mask is not None:
            # Example: if mask is (batch_size, seq_len_k), unsqueeze for n_heads and seq_len_q
            # This depends on the specific mask shape provided.
            # A common case for self-attention padding mask is (batch_size, 1, seq_len_k)
            # which needs to be (batch_size, 1, 1, seq_len_k) for ScaledDotProductAttention
            # and then broadcasted across heads.
            # If mask is (batch_size, seq_len_q, seq_len_k), it might need unsqueezing for n_heads:
            # mask = mask.unsqueeze(1) # (batch_size, 1, seq_len_q, seq_len_k)
            # The ScaledDotProductAttention expects mask like (batch_size, 1, T, T) or (B, H, T, T)
            # Let's assume the mask is already broadcastable or of shape (B, 1, T_q, T_k) or (B, 1, 1, T_k)
            pass # Mask processing logic might be needed here based on specific use case

        # 4. Scaled dot-product attention
        # context: (batch_size, n_heads, seq_len_q, d_v)
        # attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        context, attention_weights = self.attention(q_transposed, k_transposed, v_transposed, mask)

        # 5. Transpose context back: (batch_size, seq_len_q, n_heads, d_v)
        context_transposed = context.transpose(1, 2)

        # 6. Concatenate heads: (batch_size, seq_len_q, n_heads * d_v) which is (batch_size, seq_len_q, d_model)
        context_concatenated = context_transposed.contiguous().view(batch_size, -1, self.d_model)

        # 7. Output projection
        output = self.W_o(context_concatenated) # (batch_size, seq_len_q, d_model)

        # 8. Apply dropout if applicable
        if self.dropout is not None:
            output = self.dropout(output)

        return output, attention_weights
