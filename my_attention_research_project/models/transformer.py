import torch
import torch.nn as nn
import copy
import math # For sqrt if used for scaling embeddings

from .common_layers import PositionalEncoding, PositionwiseFeedForward, TokenEmbedding
# Specific attention modules will be imported dynamically within TransformerEncoder


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    (Content of TransformerEncoderLayer remains unchanged from your provided file - no conflicts here)
    """
    def __init__(self, d_model: int, attention_module: nn.Module, feed_forward_module: nn.Module, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.attention = attention_module
        self.ffn = feed_forward_module
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) 
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_normed = self.norm1(src)
        att_output_tuple = self.attention(query=src_normed, key=src_normed, value=src_normed, mask=src_mask)
        
        if isinstance(att_output_tuple, tuple):
            att_output = att_output_tuple[0] 
        else:
            att_output = att_output_tuple

        src = src + self.dropout(att_output) 
        src_normed = self.norm2(src)
        ffn_output = self.ffn(src_normed)
        src = src + self.dropout(ffn_output) 
        return src

class TransformerEncoder(nn.Module):
    """
    The Transformer Encoder, composed of an embedding layer, positional encoding,
    a stack of N identical TransformerEncoderLayers, and an output linear layer.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 attention_config: dict,
                 num_encoder_layers: int,
                 ffn_dim_factor: int = 4,
                 dropout_rate: float = 0.1,
                 use_positional_embeddings: bool = True,
                 pos_encoding_max_len: int = 5000,
                 embedding_scale_factor: float = None,
                 embedding_scale_grad_by_freq: bool = False):
        super().__init__()

        self.d_model = d_model
        self.use_positional_embeddings = use_positional_embeddings

        self.token_embedding = TokenEmbedding(
            vocab_size, 
            d_model, 
            scale_grad_by_freq=embedding_scale_grad_by_freq, 
            scale_factor=embedding_scale_factor
        )
        self.positional_encoding = PositionalEncoding(
            d_model, 
            pos_encoding_max_len, 
            dropout_rate
        )
        self.embedding_dropout = nn.Dropout(dropout_rate)
        d_ff = d_model * ffn_dim_factor

        attention_type = attention_config.get("type", "VanillaMHA")
        attention_module_instance = None
        
        if attention_type == "VanillaMHA":
            try:
                from .attention.vanilla_mha import MultiHeadAttention
            except ImportError as e:
                raise ImportError(
                    f"Failed to import MultiHeadAttention for attention_type 'VanillaMHA'. "
                    f"Ensure 'vanilla_mha.py' exists and is error-free. Original error: {e}"
                )
            if "n_heads" not in attention_config:
                raise ValueError("n_heads is required in attention_config for VanillaMHA")
            mha_n_heads = attention_config["n_heads"]
            mha_dropout_rate = attention_config.get("dropout_rate", dropout_rate) 
            attention_module_instance = MultiHeadAttention(
                d_model=d_model,
                n_heads=mha_n_heads,
                dropout_rate=mha_dropout_rate
            )
        elif attention_type == "AHC":
            try:
                from .attention.ahc_attention import AHCAttention
            except ImportError as e:
                raise ImportError(
                    f"Failed to import AHCAttention for attention_type 'AHC'. "
                    f"Ensure 'ahc_attention.py' exists and is error-free. Original error: {e}"
                )
            ahc_constructor_params = {k: v for k, v in attention_config.items() if k != 'type'}
            required_ahc_keys = {'n_heads', 'chunk_size', 'summarization_method_config', 
                                 'global_attention_method_config', 'combination_method_config'}
            missing_keys = required_ahc_keys - set(ahc_constructor_params.keys())
            if missing_keys:
                raise ValueError(f"AHC configuration is missing required keys: {missing_keys}. Provided config: {attention_config}")
            for key in ['summarization_method_config', 'global_attention_method_config', 'combination_method_config']:
                config_dict = ahc_constructor_params[key]
                if not isinstance(config_dict, dict):
                    raise ValueError(f"AHC config's '{key}' must be a dictionary. Provided: {config_dict}")
                if 'type' not in config_dict:
                        raise ValueError(f"AHC config's '{key}' must have a 'type' specified. Provided: {config_dict}")
            configured_chunk_size = ahc_constructor_params.get('chunk_size')
            if not isinstance(configured_chunk_size, int):
                raise ValueError(f"AHC config: 'chunk_size' must be an integer. Got {configured_chunk_size}")
            if configured_chunk_size <= 0:
                raise ValueError(f"AHC config: 'chunk_size' must be positive. Got {configured_chunk_size}")
            if configured_chunk_size > self.positional_encoding.max_len:
                raise ValueError(
                    f"AHC config: chunk_size ({configured_chunk_size}) "
                    f"cannot be greater than model's pos_encoding_max_len ({self.positional_encoding.max_len})."
                )
            ahc_constructor_params.setdefault('dropout_rate', dropout_rate)
            ahc_constructor_params['model_max_length'] = self.positional_encoding.max_len
            attention_module_instance = AHCAttention(
                d_model=d_model, 
                **ahc_constructor_params
            )
        elif attention_type == "PAS": # Logic from pas-attention branch
            try:
                from .attention.pas_attention import PASAttention
            except ImportError as e:
                raise ImportError(
                    f"Failed to import PASAttention for attention_type 'PAS'. "
                    f"Ensure 'pas_attention.py' exists and is error-free. Original error: {e}"
                )
            
            scanner_config = attention_config.get("scanner_config")
            focused_attention_config = attention_config.get("focused_attention_config")
            pas_dropout_rate = attention_config.get("dropout_rate", dropout_rate)

            if scanner_config is None:
                raise ValueError("scanner_config is required for PASAttention.")
            if focused_attention_config is None:
                raise ValueError("focused_attention_config is required for PASAttention.")

            attention_module_instance = PASAttention(
                d_model=d_model,
                scanner_config=scanner_config,
                focused_attention_config=focused_attention_config,
                dropout_rate=pas_dropout_rate
            )
        elif attention_type == "MOAE": # Logic from pas-attention branch (assuming it included MOAE)
            try:
                from .attention.moae_attention import MOAEAttention
            except ImportError as e:
                raise ImportError(
                    f"Failed to import MOAEAttention for attention_type 'MOAE'. "
                    f"Ensure 'moae_attention.py' exists and is error-free. Original error: {e}"
                )

            expert_configs = attention_config.get("expert_configs")
            gating_config = attention_config.get("gating_config")
            moae_dropout_rate = attention_config.get("dropout_rate", dropout_rate)

            if not isinstance(expert_configs, list) or not expert_configs:
                raise ValueError("expert_configs (list of dicts) is required for MOAEAttention and cannot be empty.")
            if not isinstance(gating_config, dict):
                raise ValueError("gating_config (dict) is required for MOAEAttention.")
            
            attention_module_instance = MOAEAttention(
                d_model=d_model,
                expert_configs=expert_configs,
                gating_config=gating_config,
                dropout_rate=moae_dropout_rate
            )
        else:
            raise ValueError(f"Unsupported attention_type in config: '{attention_type}'")

        if attention_module_instance is None:
            raise RuntimeError(f"Attention module was not instantiated for attention_type: {attention_type}")

        feed_forward_module = PositionwiseFeedForward(d_model, d_ff, dropout_rate=dropout_rate)

        encoder_layer = TransformerEncoderLayer(
            d_model,
            attention_module_instance,
            feed_forward_module,
            dropout_rate=dropout_rate
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        if self.use_positional_embeddings:
            x = self.positional_encoding(x)
        x = self.embedding_dropout(x)
        
        src_processed_mask = attention_mask # Pass the (B,S) padding mask directly

        output = x
        for layer in self.layers:
            output = layer(output, src_mask=src_processed_mask)

        output = self.norm(output)
        logits = self.output_head(output)
        return logits