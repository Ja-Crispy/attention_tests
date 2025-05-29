import torch
import torch.nn as nn
import torch.nn.functional as F # For softmax
from my_attention_research_project.models.attention.vanilla_mha import MultiHeadAttention
# Placeholder for other expert imports like PASAttention if MOAE were to support them
# from .pas_attention import PASAttention 

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
        """
        Initializes the SoftGateMLP module.

        Args:
            embed_dim: The dimensionality of the input token embeddings.
            num_experts: The number of experts to gate over.
            hidden_dim: The dimensionality of the hidden layer in the MLP.
                        If None, defaults to embed_dim.
            dropout_rate: Dropout rate to be applied within the MLP.
        """
        super().__init__()
        self.num_experts = num_experts
        
        if hidden_dim is None:
            hidden_dim = embed_dim
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SoftGateMLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            gate_weights: Tensor of shape (batch_size, seq_len, num_experts),
                          representing the probability distribution over experts for each token.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gate_weights = F.softmax(x, dim=-1)
        return gate_weights

class MOAEAttention(nn.Module):
    """
    Implements the Mixture of Attention Experts (MOAE) mechanism.
    This class will route inputs to different attention 'expert' modules
    and combine their outputs using a gating mechanism.
    """
    def __init__(self, 
                 d_model: int, 
                 expert_configs: list[dict], 
                 gating_config: dict, 
                 dropout_rate: float = 0.0):
        """
        Initializes the MOAEAttention module.
        """
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
                mha_expert = MultiHeadAttention(self.d_model, num_heads=n_heads, dropout_rate=expert_dropout_rate)
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
                mha_expert = MultiHeadAttention(self.d_model, num_heads=n_heads, dropout_rate=expert_dropout_rate)
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

            if num_experts == 0 and not self.expert_configs : # Allow no experts if expert_configs was empty
                 pass # gating_network remains None if no experts from the start
            elif num_experts == 0 and self.expert_configs : # This case means all expert configs failed to init
                 raise ValueError("Gating network cannot be created as no experts were successfully initialized.")
            elif num_experts > 0:
                self.gating_network = SoftGateMLP(
                    embed_dim=self.d_model,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim, 
                    dropout_rate=gate_dropout_rate
                )
        elif gating_type is None:
             if self.expert_configs: # If experts were intended but no gating specified
                raise ValueError("Gating type not specified in gating_config, but experts are present.")
             # If no experts and no gating type, it's fine, gating_network is None
        else:
            raise NotImplementedError(f"Gating type '{gating_type}' not supported.")


    def _generate_local_window_mask(self, 
                                   seq_len: int, 
                                   window_size: int, 
                                   device: torch.device, 
                                   batch_size: int = 1, 
                                   input_padding_mask: torch.Tensor = None) -> torch.Tensor:
        look_backward = window_size // 2
        look_forward = window_size - 1 - look_backward
        local_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - look_backward)
            end = min(seq_len, i + look_forward + 1)
            local_mask[i, start:end] = True
        local_mask = local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if input_padding_mask is not None:
            if input_padding_mask.dim() != 2 or input_padding_mask.shape[0] != batch_size or input_padding_mask.shape[1] != seq_len:
                raise ValueError(f"input_padding_mask shape must be ({batch_size}, {seq_len}), but got {input_padding_mask.shape}")
            padding_mask_bool = input_padding_mask.bool() if input_padding_mask.dtype != torch.bool else input_padding_mask
            expanded_padding_mask = padding_mask_bool.unsqueeze(1).unsqueeze(2)
            final_mask = local_mask & expanded_padding_mask
        else:
            final_mask = local_mask
        return final_mask

    def _generate_dilated_mask(self, 
                               seq_len: int, 
                               dilation_rate: int, 
                               device: torch.device, 
                               batch_size: int = 1, 
                               input_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if not isinstance(dilation_rate, int) or dilation_rate < 1:
            raise ValueError("Dilation rate must be an integer >= 1")
        dilated_key_indices = torch.arange(0, seq_len, dilation_rate, device=device, dtype=torch.long)
        key_mask_1d = torch.zeros(seq_len, device=device, dtype=torch.bool)
        key_mask_1d[dilated_key_indices] = True
        dilated_mask = key_mask_1d.unsqueeze(0).expand(seq_len, seq_len) 
        dilated_mask = dilated_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if input_padding_mask is not None:
            if input_padding_mask.dim() != 2 or input_padding_mask.shape[0] != batch_size or input_padding_mask.shape[1] != seq_len:
                raise ValueError(f"input_padding_mask shape must be ({batch_size}, {seq_len}), but got {input_padding_mask.shape}")
            padding_mask_bool = input_padding_mask.bool() if input_padding_mask.dtype != torch.bool else input_padding_mask
            expanded_padding_mask = padding_mask_bool.unsqueeze(1).unsqueeze(2)
            final_mask = dilated_mask & expanded_padding_mask
        else:
            final_mask = dilated_mask
        return final_mask


    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                input_padding_mask: torch.Tensor = None) -> torch.Tensor:
        
        if not self.experts:
            # If expert_configs was empty from the start, this is a valid "no-op" or identity if d_model matches.
            # However, typically MOAE implies experts. If it was configured to have experts but none initialized,
            # __init__ should have ideally caught it.
            # For now, let's assume if self.experts is empty, it means no experts were configured.
            # Depending on desired behavior, could return query or raise error.
            # The prompt implies error if configured but empty.
             raise RuntimeError("MOAEAttention has no experts configured.")

        if self.gating_network is None:
            # This implies either gating_type was None and no experts, or failed init.
            # If experts exist but no gating, that's an issue.
            raise RuntimeError("MOAEAttention has no gating network configured.")

        batch_size, seq_len_q, _ = query.shape
        _ , seq_len_k, _ = key.shape # key sequence length for masks
        device = query.device

        expert_outputs = []
        for expert_module in self.experts:
            expert_type = expert_module.expert_type
            current_expert_mask = None

            if expert_type == "local_window_mha":
                current_expert_mask = self._generate_local_window_mask(
                    seq_len_k, expert_module.window_size, device, batch_size, input_padding_mask)
            elif expert_type == "dilated_mha":
                current_expert_mask = self._generate_dilated_mask(
                    seq_len_k, expert_module.dilation_rate, device, batch_size, input_padding_mask)
            # Add elif blocks here for other expert types like "PAS" if they were fully integrated
            # elif expert_type == "PAS":
            #    # PASAttention might not use a pre-generated mask in the same way,
            #    # or it might take input_padding_mask directly.
            #    # Its call signature is (query, key, value, input_padding_mask)
            #    # And it returns (context_vector, hotspot_indices)
            #    expert_context, _ = expert_module(query, key, value, input_padding_mask=input_padding_mask)
            #    expert_outputs.append(expert_context)
            #    continue # Skip common MHA call below
            else:
                raise NotImplementedError(f"Mask generation or call for expert type {expert_type} not implemented in forward pass.")
            
            # Assuming MHA-like experts that return (context, weights)
            expert_context, _ = expert_module(query, key, value, mask=current_expert_mask)
            expert_outputs.append(expert_context)
        
        if not expert_outputs: # Should be caught by self.experts check, but as a safeguard
            raise RuntimeError("No expert outputs were generated.")

        # Stack expert outputs: (B, L_q, NumExperts, D)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=2) 
        
        # Get Gate Weights: (B, L_q, NumExperts)
        gate_weights = self.gating_network(query) 
        
        # Combine Expert Outputs using einsum
        # gate_weights: (B, L_q, NumExperts) -> 'bln'
        # stacked_expert_outputs: (B, L_q, NumExperts, D) -> 'blnd'
        # combined_context: (B, L_q, D) -> 'bld'
        combined_context = torch.einsum('bln,blnd->bld', gate_weights, stacked_expert_outputs)
        
        return combined_context
