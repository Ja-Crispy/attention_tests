import torch
import torch.nn as nn
import torch.nn.functional as F
from my_attention_research_project.models.attention.vanilla_mha import MultiHeadAttention

class PASAttention(nn.Module):
    """
    Implements the Predictive Attention Scaffolding (PAS) mechanism.
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
        scanner_type = self.scanner_config.get("type")
        if scanner_type == "PAS_P1_linear":
            self.scanner_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.scanner_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.top_k_hotspots = self.scanner_config.get("top_k_hotspots")
            if self.top_k_hotspots is None:
                raise ValueError("top_k_hotspots must be provided in scanner_config for PAS_P1_linear")
            if not isinstance(self.top_k_hotspots, int) or self.top_k_hotspots < 0:
                raise ValueError("top_k_hotspots must be a non-negative integer.")
            self.scanner = self._linear_scanner
        # Add elif for "PAS_P2_conv" here when implemented
        else:
            raise NotImplementedError(f"Scanner type '{scanner_type}' is not implemented or not configured.")

        # Focused Attention MHA initialization
        mha_num_heads = self.focused_attention_config.get("num_heads")
        if mha_num_heads is None:
            raise ValueError("num_heads must be provided in focused_attention_config")
        
        # Use dropout_rate from this module's init as default if not in focused_attention_config
        mha_dropout_rate = self.focused_attention_config.get("dropout_rate", dropout_rate) 
        
        self.focused_attention_mha = MultiHeadAttention(
            d_model=self.d_model, 
            n_heads=mha_num_heads, # Corrected parameter name
            dropout_rate=mha_dropout_rate
        )
        # Assign the method directly if only one type of focused attention is planned
        # If multiple types were possible, a factory similar to scanner could be used.
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
        # Using ELU + 1 as a simple non-negative kernel approximation
        scores = F.elu(scores) + 1 

        if key_padding_mask is not None:
            # key_padding_mask is (B, S_k), True for valid. MHF expects False for mask.
            mask_for_fill = key_padding_mask.unsqueeze(1) # (B, 1, S_k)
            if mask_for_fill.dtype != torch.bool: 
                mask_for_fill = mask_for_fill.bool() # Ensure boolean for masked_fill
            # Where mask_for_fill is False (i.e., original padding), set scores to -inf
            scores = scores.masked_fill(mask_for_fill == False, float('-inf'))

        # Determine actual k for topk, ensuring it's not larger than available keys
        # and handles case where key_embed might be empty along seq_len_k
        key_seq_len = scores.size(-1)
        current_k = min(self.top_k_hotspots, key_seq_len)
        
        if key_seq_len == 0 : # No keys to select from
            batch_size, seq_len_q, _ = query_embed.shape
            # Return empty indices tensor of correct shape for top_k=0
            return torch.empty((batch_size, seq_len_q, 0), dtype=torch.long, device=query_embed.device)

        if current_k == 0: # If top_k_hotspots is 0
             batch_size, seq_len_q, _ = query_embed.shape
             return torch.empty((batch_size, seq_len_q, 0), dtype=torch.long, device=query_embed.device)
        
        _, hotspot_indices = torch.topk(scores, k=current_k, dim=-1, sorted=False) # Not necessarily sorted
        
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
            hotspot_indices: Indices from Stage 1 (batch_size, seq_len_q, top_k).
            original_key_padding_mask: Padding mask for the original key sequence (batch_size, seq_len_k_orig).
                                       True/1 for valid, False/0 for pad.
        Returns:
            context_vector: Output from the focused attention (batch_size, seq_len_q, d_model).
        """
        batch_size, seq_len_q, d_model_q = query.shape
        _, _, top_k = hotspot_indices.shape # top_k is the actual number of gathered indices

        if top_k == 0: # No hotspots selected or available
            return torch.zeros_like(query)

        # Gather K and V based on hotspot_indices
        # hotspot_indices: (B, S_q, top_k)
        # key: (B, S_k_orig, D) -> expand for S_q -> (B, S_q, S_k_orig, D)
        # expanded_hotspot_indices for gather: (B, S_q, top_k, D)
        expanded_hotspot_indices_for_gather = hotspot_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_model)

        k_gathered = torch.gather(key.unsqueeze(1).expand(-1, seq_len_q, -1, -1), 
                                  dim=2, 
                                  index=expanded_hotspot_indices_for_gather)
        v_gathered = torch.gather(value.unsqueeze(1).expand(-1, seq_len_q, -1, -1), 
                                  dim=2, 
                                  index=expanded_hotspot_indices_for_gather)
        # k_gathered, v_gathered: (B, S_q, top_k, D)

        # Create the mask for the sparse attention (MHA expects mask where 0 means mask)
        # This mask should be (B, S_q, top_k) where True/1 means valid to attend.
        # It needs to account for padding in the *original* key sequence for the gathered items.
        gathered_attention_mask = torch.ones(batch_size, seq_len_q, top_k, device=query.device, dtype=torch.bool)
        if original_key_padding_mask is not None:
            # original_key_padding_mask: (B, S_k_orig), True for valid
            # We need to gather the padding status of the selected hotspots
            # gathered_padding_status: (B, S_q, top_k)
            gathered_padding_status = torch.gather(
                original_key_padding_mask.unsqueeze(1).expand(-1, seq_len_q, -1), 
                dim=2, 
                index=hotspot_indices
            )
            gathered_attention_mask = gathered_attention_mask & gathered_padding_status
        
        # Reshape query and gathered K,V for batched MHA
        # Each (query_token_i, its_top_k_keys, its_top_k_values) will be a separate item in a larger batch for MHA
        # Query: (B, S_q, D) -> (B * S_q, 1, D)
        q_reshaped = query.contiguous().view(batch_size * seq_len_q, 1, self.d_model)
        # Keys: (B, S_q, top_k, D) -> (B * S_q, top_k, D)
        k_reshaped = k_gathered.contiguous().view(batch_size * seq_len_q, top_k, self.d_model)
        v_reshaped = v_gathered.contiguous().view(batch_size * seq_len_q, top_k, self.d_model)

        # Prepare mask for MHA: (B, S_q, top_k) -> (B * S_q, 1, 1, top_k)
        # This is a padding mask for the gathered keys. MHA expects 0 for masked.
        mha_mask = gathered_attention_mask.view(batch_size * seq_len_q, 1, top_k)
        mha_mask = mha_mask.unsqueeze(1) # (B*S_q, 1, 1, top_k)
        if mha_mask.dtype != torch.bool:
            mha_mask = mha_mask.bool() # Ensure boolean for masked_fill in MHA

        context_reshaped, _ = self.focused_attention_mha(q_reshaped, k_reshaped, v_reshaped, mask= (mha_mask == False) ) # MHA expects False to mask
        
        context_vector = context_reshaped.view(batch_size, seq_len_q, self.d_model)
        return context_vector

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                input_padding_mask: torch.Tensor = None): # Using input_padding_mask for clarity
        """
        Performs the forward pass of the PASAttention module.

        Args:
            query: The query tensor (batch_size, seq_len_q, d_model).
            key: The key tensor (batch_size, seq_len_k_orig, d_model).
            value: The value tensor (batch_size, seq_len_k_orig, d_model).
            input_padding_mask: An optional mask for input padding of the original key sequence (batch_size, seq_len_k_orig).
                                Assumed to be True/1 for valid tokens, False/0 for padding.

        Returns:
            context_vector: The output tensor from the focused attention mechanism.
            hotspot_indices (torch.Tensor, optional): Indices of the top_k hotspots. Returned for debugging/analysis.
        """
        hotspot_indices = None
        if self.scanner is not None:
            # The scanner needs the key_padding_mask which is our input_padding_mask
            hotspot_indices = self.scanner(query, key, input_padding_mask)
        else:
            # This case should ideally be caught by __init__ if scanner_type is not supported
            raise NotImplementedError("Scanner is not configured or scanner type is invalid, but PAS requires a scanner.")

        if self.focused_attention is not None:
            # Pass the original input_padding_mask (which corresponds to the original key sequence)
            # to _focused_sparse_attention so it can correctly mask the gathered keys.
            context_vector = self.focused_attention(query, key, value, hotspot_indices, input_padding_mask)
            return context_vector, hotspot_indices # Returning hotspot_indices for potential analysis
        else:
            # This case should also be caught by __init__
            raise NotImplementedError("Focused attention is not configured.")