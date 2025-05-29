import torch
import torch.nn as nn
import torch.nn.functional as F # For softmax in SoftGateMLP or forward
from my_attention_research_project.models.attention.vanilla_mha import MultiHeadAttention
# Placeholder for other expert imports like PASAttention if MOAE were to support them
# from .pas_attention import PASAttention

# Taking SoftGateMLP from the 'pas-attention' branch as it's a good modular component
class SoftGateMLP(nn.Module):
    """
    A simple MLP-based soft gating mechanism for MOAE.
    It takes token embeddings and produces a probability distribution over experts for each token.
    """
    def __init__(self,
                 embed_dim: int,
                 num_experts: int,
                 hidden_dim: int = None,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gate_weights = F.softmax(x, dim=-1) # Softmax to get probabilities
        return gate_weights

class MOAEAttention(nn.Module):
    """
    Implements the Mixture of Attention Experts (MOAE) mechanism.
    """
    def __init__(self,
                 d_model: int,
                 expert_configs: list[dict],
                 gating_config: dict,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.expert_configs = expert_configs
        self.gating_config = gating_config
        self.dropout_rate = dropout_rate

        self.experts = nn.ModuleList()
        for exp_conf in self.expert_configs:
            expert_type = exp_conf.get("type")
            if expert_type == "local_window_mha":
                n_heads = exp_conf.get("n_heads")
                if n_heads is None:
                    raise ValueError("n_heads is required for local_window_mha expert.")
                window_size = exp_conf.get("window_size")
                if window_size is None:
                    raise ValueError("window_size is required for local_window_mha expert.")
                expert_dropout_rate = exp_conf.get("dropout_rate", self.dropout_rate)
                # Using n_heads from initial-project-setup
                mha_expert = MultiHeadAttention(self.d_model, n_heads=n_heads, dropout_rate=expert_dropout_rate)
                mha_expert.expert_type = "local_window_mha"
                mha_expert.window_size = window_size
                self.experts.append(mha_expert)

            elif expert_type == "dilated_mha":
                n_heads = exp_conf.get("n_heads")
                if n_heads is None:
                    raise ValueError("n_heads is required for dilated_mha expert.")
                dilation_rate = exp_conf.get("dilation_rate")
                if dilation_rate is None:
                    raise ValueError("dilation_rate is required for dilated_mha expert.")
                if not isinstance(dilation_rate, int) or dilation_rate < 1:
                    raise ValueError("dilation_rate must be an integer >= 1.")
                expert_dropout_rate = exp_conf.get("dropout_rate", self.dropout_rate)
                # Using n_heads from initial-project-setup
                mha_expert = MultiHeadAttention(self.d_model, n_heads=n_heads, dropout_rate=expert_dropout_rate)
                mha_expert.expert_type = "dilated_mha"
                mha_expert.dilation_rate = dilation_rate
                self.experts.append(mha_expert)
            else:
                raise ValueError(f"Unsupported expert type: {expert_type} in expert_configs.")

        # Instantiate Gating Network
        gating_type = self.gating_config.get("type")
        self.gating_network = None

        if gating_type == "soft_gate_mlp":
            hidden_dim = self.gating_config.get("hidden_dim")
            gate_dropout_rate = self.gating_config.get("dropout_rate", self.dropout_rate)
            num_experts = len(self.experts)

            if num_experts == 0:
                if self.expert_configs: # Configured to have experts, but none initialized
                    raise ValueError("Gating network cannot be created as no experts were successfully initialized.")
                # else: no experts configured, gating_network remains None, which is fine.
            else: # num_experts > 0
                self.gating_network = SoftGateMLP(
                    embed_dim=self.d_model,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    dropout_rate=gate_dropout_rate
                )
        elif gating_type is None:
            if self.expert_configs: # Experts were intended but no gating specified
                raise ValueError("Gating type not specified in gating_config, but experts are present.")
        else:
            raise NotImplementedError(f"Gating type '{gating_type}' not supported.")

    # Using mask generation from 'pas-attention' branch as they are more detailed
    # and handle input_padding_mask to produce a 4D mask.
    # The MHA module should be able to handle a 4D mask (B, 1, S, S) or (B, H, S, S).
    # VanillaMHA's ScaledDotProductAttention takes (B, H, Q_len, K_len) or broadcastable, so (B,1,S,S) is fine.
    def _generate_local_window_mask(self,
                                   seq_len: int,
                                   window_size: int,
                                   device: torch.device,
                                   batch_size: int = 1,
                                   input_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # This creates a causal mask within the window
        look_backward = window_size // 2
        look_forward = window_size - 1 - look_backward
        
        # Create a mask for local window, allow attending to current and past/future within window
        # This is NOT strictly causal yet. Causal part is combined.
        local_window_indices = torch.arange(seq_len, device=device).unsqueeze(0) # (1, S)
        col_indices = local_window_indices.unsqueeze(1) # (S, 1) -> for broadcasting for query
        row_indices = local_window_indices # (1, S) -> for broadcasting for key
        
        # Mask where |col_idx - row_idx| is within window limits
        # For a query at col_idx, it can attend to keys at row_idx where:
        # col_idx - look_backward <= row_idx <= col_idx + look_forward
        mask = (row_indices >= (col_indices - look_backward)) & (row_indices <= (col_indices + look_forward))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len) # (B, 1, S, S)

        # Combine with causal mask: ensure attention is only to past and current
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        final_mask = mask & causal_mask

        if input_padding_mask is not None: # (B, S_key)
            if input_padding_mask.dim() != 2 or input_padding_mask.shape[0] != batch_size or input_padding_mask.shape[1] != seq_len:
                raise ValueError(f"input_padding_mask shape must be ({batch_size}, {seq_len}), but got {input_padding_mask.shape}")
            padding_mask_bool = input_padding_mask.bool() if input_padding_mask.dtype != torch.bool else input_padding_mask
            # Expand padding mask for key dimension: (B, S_k) -> (B, 1, 1, S_k)
            expanded_padding_mask = padding_mask_bool.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
            final_mask = final_mask & expanded_padding_mask # Apply padding mask
        
        return final_mask # True for attend, False for mask

    def _generate_dilated_mask(self,
                               seq_len: int,
                               dilation_rate: int,
                               device: torch.device,
                               batch_size: int = 1,
                               input_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if not isinstance(dilation_rate, int) or dilation_rate < 1:
            raise ValueError("Dilation rate must be an integer >= 1")

        # Query can attend to key if key_idx is a multiple of dilation_rate AND key_idx <= query_idx (causal)
        query_indices = torch.arange(seq_len, device=device).unsqueeze(1) # (S_q, 1)
        key_indices = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, S_k)

        # Dilated condition: key_indices must be one of the dilated positions
        is_dilated_key = (key_indices % dilation_rate) == 0
        
        # Causal condition
        is_causal = key_indices <= query_indices
        
        final_mask = is_dilated_key & is_causal # (S_q, S_k)
        final_mask = final_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len) # (B, 1, S, S)

        if input_padding_mask is not None: # (B, S_key)
            if input_padding_mask.dim() != 2 or input_padding_mask.shape[0] != batch_size or input_padding_mask.shape[1] != seq_len:
                raise ValueError(f"input_padding_mask shape must be ({batch_size}, {seq_len}), but got {input_padding_mask.shape}")
            padding_mask_bool = input_padding_mask.bool() if input_padding_mask.dtype != torch.bool else input_padding_mask
            expanded_padding_mask = padding_mask_bool.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S_k)
            final_mask = final_mask & expanded_padding_mask # Apply padding for keys
            
        return final_mask # True for attend, False for mask

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                # Taking 'input_padding_mask' from 'pas-attention' as it's more descriptive for a 2D mask
                input_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = query.shape
        device = query.device

        if not self.experts:
            # This was "raise RuntimeError" in pas-attention, but initial-project-setup returns zeros.
            # Returning zeros might be safer if this state is possible and needs graceful handling.
            # However, __init__ validation should prevent this if experts were configured but failed.
            # If no experts were configured (empty expert_configs), this is okay.
            if self.expert_configs : # Experts were configured but list is empty
                 raise RuntimeError("MOAEAttention: Experts configured but none were initialized.")
            # No experts configured: return query itself or zeros. For attention, zeros is safer.
            return torch.zeros_like(query), torch.zeros(batch_size, 0, seq_len, device=device) # 0 experts

        if self.gating_network is None:
             # This implies gating_type was None and no experts, or failed init.
             # If experts exist but no gating, __init__ should have caught it.
             if self.experts: # Experts exist, but no gating network (e.g. gating_type was None in config)
                 raise RuntimeError("MOAEAttention: Experts are present, but no gating network is configured/initialized.")
             # If no experts and no gating network (both intended to be None), this is like the above "no experts" case.
             return torch.zeros_like(query), torch.zeros(batch_size, 0, seq_len, device=device)


        # Get Gate Weights from SoftGateMLP: (B, L_q, NumExperts)
        # The MLP already applies softmax.
        gate_weights_or_probs = self.gating_network(query)

        expert_outputs = []
        # Storing all head attention weights from experts is memory intensive.
        # For now, let's just return gating weights as a proxy for "attention weights" of MOAE.
        # Individual expert attention weights could be collected if truly needed for deep analysis.

        for expert_module in self.experts:
            expert_type = expert_module.expert_type
            current_expert_mask_4D = None # MHA expects mask where False means "mask this position"

            if expert_type == "local_window_mha":
                # _generate_local_window_mask returns True for attend
                current_expert_mask_4D = self._generate_local_window_mask(
                    seq_len, expert_module.window_size, device, batch_size, input_padding_mask)
            elif expert_type == "dilated_mha":
                # _generate_dilated_mask returns True for attend
                current_expert_mask_4D = self._generate_dilated_mask(
                    seq_len, expert_module.dilation_rate, device, batch_size, input_padding_mask)
            else:
                # Fallback or error for unhandled expert types that might need specific masks
                # If an expert is a plain VanillaMHA, it might just use the input_padding_mask transformed.
                # For now, assume all configured experts have specific mask logic here.
                 raise NotImplementedError(f"Mask generation for expert type {expert_type} not handled in forward.")

            # MHA expects mask == 0 to mask. So, invert the boolean mask.
            mha_compat_mask = (current_expert_mask_4D == False) if current_expert_mask_4D is not None else None
            
            expert_context, _ = expert_module(query, key, value, mask=mha_compat_mask)
            expert_outputs.append(expert_context)
        
        if not expert_outputs: # Should be caught by self.experts check earlier
            raise RuntimeError("No expert outputs were generated despite having experts.")

        # Stack expert outputs: (B, L_q, NumExperts, D)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # Combine Expert Outputs using einsum (from pas-attention HEAD)
        # gate_weights_or_probs: (B, L_q, NumExperts) -> 'bln'
        # stacked_expert_outputs: (B, L_q, NumExperts, D) -> 'blnd'
        # combined_context: (B, L_q, D) -> 'bld'
        combined_context = torch.einsum('bln,blnd->bld', gate_weights_or_probs, stacked_expert_outputs)
        
        # Return gating weights as the "attention_weights" for MOAE for monitoring/analysis
        # Shape (B, NumExperts, L_q)
        gating_weights_to_return = gate_weights_or_probs.permute(0, 2, 1)
        
        return combined_context, gating_weights_to_return