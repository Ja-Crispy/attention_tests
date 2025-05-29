import torch
import torch.nn as nn
import torch.nn.functional as F
from my_attention_research_project.models.attention.vanilla_mha import MultiHeadAttention

class PASAttention(nn.Module):
    """
    Implements the Perceiver Attention Sampling (PAS) mechanism.
    """
    def __init__(self, 
                 d_model: int, 
                 scanner_config: dict, 
                 focused_attention_config: dict, 
                 dropout_rate: float = 0.0): # This dropout_rate is a general default
        """
        Initializes the PASAttention module.

        Args:
            d_model: The dimensionality of the input and output features.
            scanner_config: Configuration dictionary for the scanner module.
            focused_attention_config: Configuration dictionary for the focused attention module.
            dropout_rate: Default dropout rate if not specified in focused_attention_config.
        """
        super().__init__()
        self.d_model = d_model
        self.scanner_config = scanner_config
        self.focused_attention_config = focused_attention_config
        # self.dropout_rate = dropout_rate # General dropout, not directly used here but stored if needed elsewhere

        # Scanner initialization
        if self.scanner_config.get("type") == "PAS_P1_linear":
            self.scanner_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.scanner_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.top_k_hotspots = self.scanner_config.get("top_k_hotspots")
            if self.top_k_hotspots is None:
                raise ValueError("top_k_hotspots must be provided in scanner_config for PAS_P1_linear")
            self.scanner = self._linear_scanner
        else:
            self.scanner = None 

        # Focused Attention MHA initialization
        mha_num_heads = self.focused_attention_config.get("num_heads")
        if mha_num_heads is None:
            raise ValueError("num_heads must be provided in focused_attention_config")
        
        # Use 0.0 as default for MHA dropout_rate if not specified in its config
        mha_dropout_rate = self.focused_attention_config.get("dropout_rate", 0.0) 
        
        self.focused_attention_mha = MultiHeadAttention(
            d_model=self.d_model, 
            n_heads=mha_num_heads, 
            dropout_rate=mha_dropout_rate
        )
        self.focused_attention = self._focused_sparse_attention


    def _linear_scanner(self, 
                        query_embed: torch.Tensor, 
                        key_embed: torch.Tensor, 
                        key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the linear scan to identify hotspots.
        Args:
            query_embed: Query tensor (batch_size, seq_len_q, d_model).
            key_embed: Key tensor (batch_size, seq_len_k, d_model).
            key_padding_mask: Padding mask for keys (batch_size, seq_len_k),
                              where True or 1.0 indicates valid tokens, False or 0.0 for padding.
        Returns:
            hotspot_indices: Indices of the top_k hotspots (batch_size, seq_len_q, top_k_hotspots).
        """
        q_proj = self.scanner_q_proj(query_embed)
        k_proj = self.scanner_k_proj(key_embed)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        scores = F.elu(scores) + 1

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1) # (batch_size, 1, seq_len_k)
            if mask.dtype != torch.bool: 
                mask = mask.bool()
            scores = scores.masked_fill(mask == False, float('-inf'))

        k_dim_size = scores.size(-1)
        current_k = min(self.top_k_hotspots, k_dim_size)
        
        if k_dim_size == 0 : 
            batch_size, seq_len_q, _ = query_embed.shape
            return torch.empty((batch_size, seq_len_q, 0), dtype=torch.long, device=query_embed.device)

        # If current_k is 0 (e.g. self.top_k_hotspots is 0), topk returns empty tensors.
        # If all scores are -inf, topk still returns indices (they just correspond to -inf scores).
        # The sparse attention mask will handle these 'invalid' selections.
        _, hotspot_indices = torch.topk(scores, k=current_k, dim=-1)
        
        return hotspot_indices

    def _focused_sparse_attention(self, 
                                  query: torch.Tensor, 
                                  key: torch.Tensor, 
                                  value: torch.Tensor, 
                                  hotspot_indices: torch.Tensor, 
                                  original_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs focused sparse attention using the identified hotspots.
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model).
            key: Original full key sequence (batch_size, seq_len_k_orig, d_model).
            value: Original full value sequence (batch_size, seq_len_k_orig, d_model).
            hotspot_indices: Indices from Stage 1 (batch_size, seq_len_q, top_k_hotspots).
            original_key_padding_mask: Padding mask for the original key sequence (batch_size, seq_len_k_orig).
                                       True/1 for valid, False/0 for pad.
        Returns:
            context_vector: Output from the focused attention (batch_size, seq_len_q, d_model).
        """
        batch_size, seq_len_q, d_model_q = query.shape # d_model_q should be self.d_model
        _, _, top_k = hotspot_indices.shape

        if top_k == 0: # No hotspots selected (e.g. top_k_hotspots was 0 or seq_len_k was 0)
            return torch.zeros_like(query)

        expanded_hotspot_indices = hotspot_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)

        k_gathered = torch.gather(key.unsqueeze(1).expand(-1, seq_len_q, -1, -1), 
                                  dim=2, 
                                  index=expanded_hotspot_indices)
        v_gathered = torch.gather(value.unsqueeze(1).expand(-1, seq_len_q, -1, -1), 
                                  dim=2, 
                                  index=expanded_hotspot_indices)

        sparse_mask = torch.ones(batch_size, seq_len_q, top_k, device=query.device, dtype=torch.bool)

        if original_key_padding_mask is not None:
            # original_key_padding_mask: (B, S_k_orig), True for valid
            # hotspot_indices: (B, S_q, top_k)
            # gathered_padding_status: (B, S_q, top_k), True if original key at this hotspot index was valid
            gathered_padding_status = torch.gather(
                original_key_padding_mask.unsqueeze(1).expand(-1, seq_len_q, -1), 
                dim=2, 
                index=hotspot_indices
            )
            sparse_mask = sparse_mask & gathered_padding_status
        
        # sparse_mask: (batch_size, seq_len_q, top_k), True means attend, False means mask.
        # MHA expects mask where False means "mask this position". This is consistent.
        sparse_mask_for_mha = sparse_mask.unsqueeze(1).unsqueeze(2) # (B, 1, S_q, top_k)
        
        q_reshaped = query.contiguous().view(batch_size * seq_len_q, 1, self.d_model)
        k_reshaped = k_gathered.contiguous().view(batch_size * seq_len_q, top_k, self.d_model)
        v_reshaped = v_gathered.contiguous().view(batch_size * seq_len_q, top_k, self.d_model)
        
        # Mask for MHA: (B * S_q, 1, 1, top_k)
        # Each of the (B * S_q) queries has length 1.
        # The keys for each query have length top_k.
        # The MHA mask should be (N, num_heads, L_q, L_k) or (N, L_q, L_k)
        # Here N = B * S_q, L_q = 1, L_k = top_k
        # So, mask_reshaped should be (B * S_q, 1, top_k) or (B*S_q, num_heads, 1, top_k)
        # The prompt specifies mask_reshaped = sparse_mask_for_mha.view(batch_size * seq_len_q, 1, 1, top_k)
        mask_reshaped = sparse_mask_for_mha.view(batch_size * seq_len_q, 1, 1, top_k)

        context_reshaped, _ = self.focused_attention_mha(q_reshaped, k_reshaped, v_reshaped, mask=mask_reshaped)
        
        context_vector = context_reshaped.view(batch_size, seq_len_q, self.d_model)
        return context_vector

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        Performs the forward pass of the PASAttention module.

        Args:
            query: The query tensor (batch_size, seq_len_q, d_model).
            key: The key tensor (batch_size, seq_len_k_orig, d_model).
            value: The value tensor (batch_size, seq_len_k_orig, d_model).
            mask: An optional mask for input padding (batch_size, seq_len_k_orig).
                  Assumed to be True/1 for valid, False/0 for padding.
                  This parameter is renamed from input_padding_mask for consistency
                  with other attention modules in the TransformerEncoderLayer.

        Returns:
            context_vector: The output tensor from the focused attention mechanism.
            hotspot_indices: Indices of the top_k hotspots.
        """
        # Map mask parameter to input_padding_mask for internal use
        input_padding_mask = mask
        
        hotspot_indices = None
        if self.scanner is not None:
            hotspot_indices = self.scanner(query, key, input_padding_mask)
        else:
            raise NotImplementedError("Scanner is not configured, but PAS requires a scanner.")

        if self.focused_attention is not None:
            # Pass original_key_padding_mask (which is input_padding_mask) to focused attention
            context_vector = self.focused_attention(query, key, value, hotspot_indices, input_padding_mask)
            return context_vector, hotspot_indices
        else:
            raise NotImplementedError("Focused attention is not configured.")
