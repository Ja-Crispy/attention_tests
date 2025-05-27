import torch
import torch.nn as nn
import copy
import math # For sqrt if used for scaling embeddings

from .common_layers import PositionalEncoding, PositionwiseFeedForward, TokenEmbedding
# from .attention.vanilla_mha import MultiHeadAttention # Removed global import
# from .attention.ahc_attention import AHCAttention # Removed global import


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.

    It consists of a self-attention mechanism and a position-wise feed-forward network.
    Layer normalization and residual connections are applied around each of these sub-layers.
    This implementation uses pre-LayerNorm: Norm -> Sublayer -> Dropout -> Residual.

    Args:
        d_model (int): The number of expected features in the input (required).
        attention_module (nn.Module): An instance of an attention mechanism
                                      (e.g., MultiHeadAttention).
        feed_forward_module (nn.Module): An instance of a feed-forward network
                                         (e.g., PositionwiseFeedForward).
        dropout_rate (float): The dropout value applied to the output of each sub-layer.
    """
    def __init__(self, d_model: int, attention_module: nn.Module, feed_forward_module: nn.Module, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.attention = attention_module
        self.ffn = feed_forward_module

        # Layer normalization for the attention block (pre-norm)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) 
        # Layer normalization for the feed-forward block (pre-norm)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout for both sub-layers (applied after the sub-layer, before residual add)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the TransformerEncoderLayer.

        Args:
            src (torch.Tensor): The sequence to the encoder layer.
                                Shape: (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): The mask for the src sequence.
                                               Its shape depends on the attention mechanism.
                                               For self-attention in MHA, it's typically
                                               (batch_size, 1, 1, seq_len) for padding or
                                               (batch_size, 1, seq_len, seq_len) for combined masks.
                                               Defaults to None.

        Returns:
            torch.Tensor: The output of the encoder layer.
                          Shape: (batch_size, seq_len, d_model).
        """
        # 1. Attention block (Norm -> Attention -> Dropout -> Residual)
        src_normed = self.norm1(src)
        # Compute attention output. For encoder self-attention, Q, K, V are all from src_normed.
        att_output, _ = self.attention(query=src_normed, key=src_normed, value=src_normed, mask=src_mask)
        src = src + self.dropout(att_output) # Apply dropout to attention output and add residual

        # 2. Feed-forward block (Norm -> FFN -> Dropout -> Residual)
        src_normed = self.norm2(src)
        ffn_output = self.ffn(src_normed)
        src = src + self.dropout(ffn_output) # Apply dropout to FFN output and add residual

        return src

class TransformerEncoder(nn.Module):
    """
    The Transformer Encoder, composed of an embedding layer, positional encoding,
    a stack of N identical TransformerEncoderLayers, and an output linear layer.
    This version is designed to be a more complete model for tasks like Causal LM.

    Args:
        vocab_size (int): Size of the input vocabulary.
        d_model (int): The dimension of the model (embeddings, attention, etc.).
        attention_config (dict): Configuration for the attention module.
        num_encoder_layers (int): Number of stacked TransformerEncoderLayer instances.
        ffn_dim_factor (int, optional): Factor to determine d_ff (d_ff = d_model * ffn_dim_factor).
                                       Defaults to 4.
        dropout_rate (float, optional): Dropout rate used throughout the model (embeddings, attention, FFN, encoder layers).
                                       Defaults to 0.1.
        use_positional_embeddings (bool, optional): Whether to add positional encodings.
                                                  Defaults to True.
        pos_encoding_max_len (int, optional): Maximum sequence length for positional encodings.
                                            Defaults to 5000.
        embedding_scale_factor (float, optional): Factor to scale token embeddings.
                                                  If None, no scaling. If set (e.g. math.sqrt(d_model)),
                                                  embeddings are multiplied by this factor. Defaults to None.
        embedding_scale_grad_by_freq (bool, optional): Passed to TokenEmbedding for nn.Embedding's
                                                       `scale_grad_by_freq` argument. Defaults to False.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 attention_config: dict, # Added attention_config
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

        # 1. Token Embedding Layer
        self.token_embedding = TokenEmbedding(
            vocab_size, 
            d_model, 
            scale_grad_by_freq=embedding_scale_grad_by_freq, 
            scale_factor=embedding_scale_factor
        )

        # 2. Positional Encoding Layer
        # PositionalEncoding applies dropout internally after adding PE to the embeddings.
        self.positional_encoding = PositionalEncoding(
            d_model, 
            pos_encoding_max_len, 
            dropout_rate # Dropout here is applied by PositionalEncoding itself
        )
        
        # 3. Additional Dropout after embedding + PE
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # 4. Encoder Layers
        d_ff = d_model * ffn_dim_factor

        # Dynamically instantiate attention module
        attention_type = attention_config.get("type", "VanillaMHA")
        attention_module = None # Initialize local variable
        
        if attention_type == "VanillaMHA":
            try:
                from .attention.vanilla_mha import MultiHeadAttention # Dynamic import
            except ImportError as e:
                raise ImportError(
                    f"Failed to import MultiHeadAttention for attention_type 'VanillaMHA'. "
                    f"Ensure 'vanilla_mha.py' exists and is error-free. Original error: {e}"
                )
            
            if "n_heads" not in attention_config:
                raise ValueError("n_heads is required in attention_config for VanillaMHA")
            
            mha_n_heads = attention_config["n_heads"]
            mha_dropout_rate = attention_config.get("dropout_rate", dropout_rate)

            attention_module = MultiHeadAttention(
                d_model=d_model,
                n_heads=mha_n_heads,
                dropout_rate=mha_dropout_rate
            )
        elif attention_type == "AHC":
            try:
                from .attention.ahc_attention import AHCAttention # Dynamic import
            except ImportError as e:
                raise ImportError(
                    f"Failed to import AHCAttention for attention_type 'AHC'. "
                    f"Ensure 'ahc_attention.py' exists and is error-free. Original error: {e}"
                )
            
            ahc_constructor_params = {k: v for k, v in attention_config.items() if k != 'type'}
            
            required_ahc_keys = {'n_heads', 'chunk_size', 'summarization_method_config', 'global_attention_method_config', 'combination_method_config'}
            missing_keys = required_ahc_keys - set(ahc_constructor_params.keys())
            if missing_keys:
                raise ValueError(f"AHC configuration is missing required keys: {missing_keys}. Provided config: {attention_config}")

            # Further validation for nested configs (structure and 'type' key)
            for key in ['summarization_method_config', 'global_attention_method_config', 'combination_method_config']:
                config_dict = ahc_constructor_params[key] # Access directly as it's a required key
                if not isinstance(config_dict, dict):
                    raise ValueError(f"AHC config's '{key}' must be a dictionary. Provided: {config_dict}")
                if 'type' not in config_dict: # Check for 'type' key within the nested dict
                     raise ValueError(f"AHC config's '{key}' must have a 'type' specified. Provided: {config_dict}")

            configured_chunk_size = ahc_constructor_params.get('chunk_size')
            if not isinstance(configured_chunk_size, int):
                raise ValueError(f"AHC config: 'chunk_size' must be an integer. Got {configured_chunk_size}")
            if configured_chunk_size <= 0:
                raise ValueError(f"AHC config: 'chunk_size' must be positive. Got {configured_chunk_size}")
            
            model_max_len = self.positional_encoding.max_len 
            if configured_chunk_size > model_max_len:
                raise ValueError(
                    f"AHC config: chunk_size ({configured_chunk_size}) "
                    f"cannot be greater than model's pos_encoding_max_len ({model_max_len})."
                )
            
            ahc_constructor_params.setdefault('dropout_rate', dropout_rate)

            attention_module = AHCAttention(
                d_model=d_model, 
                **ahc_constructor_params
            )
        else:
            raise ValueError(f"Unsupported attention_type in config: '{attention_type}'")

        if attention_module is None: # Should not happen if logic is correct, but as a safeguard
            raise RuntimeError(f"Attention module was not instantiated for attention_type: {attention_type}")

        feed_forward_module = PositionwiseFeedForward(d_model, d_ff, dropout_rate=dropout_rate)

        # Note: The dropout_rate passed to TransformerEncoderLayer is for dropout on att_output and ffn_output
        # within that layer, before the residual connection.
        encoder_layer = TransformerEncoderLayer(
            d_model,
            attention_module,
            feed_forward_module,
            dropout_rate=dropout_rate
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        # 5. Final Layer Normalization (applied after all encoder layers)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # 6. Output Head (for language modeling tasks)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the TransformerEncoder model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
                                      Shape: (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Mask for the input sequence (1 for tokens, 0 for padding).
                                                     Shape: (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Logits over the vocabulary.
                          Shape: (batch_size, seq_len, vocab_size).
        """
        # 1. Embeddings and Positional Encoding
        x = self.token_embedding(input_ids) # Shape: (batch_size, seq_len, d_model)
        
        if self.use_positional_embeddings:
            x = self.positional_encoding(x) # PositionalEncoding applies dropout internally
        
        x = self.embedding_dropout(x) # Additional dropout after embeddings and PE

        # 2. Prepare Attention Mask
        src_mask = None
        if attention_mask is not None:
            # Input attention_mask is (batch_size, seq_len) where 1=real token, 0=padding.
            # MHA expects mask for masked_fill where 0 means "mask this token (set to -inf)".
            # So, if attention_mask is already 0 for padding, it's in the correct format.
            # We need to reshape it for MHA: (batch_size, 1, 1, seq_len) for broadcasting.
            src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Ensure mask values are boolean for masked_fill, or that the logic in ScaledDotProductAttention
            # handles 0/1 appropriately (e.g., `scores.masked_fill(mask == 0, -1e9)`).
            # If mask is float, it might need conversion: src_mask = src_mask.bool()
            # Given our MHA implementation, `mask == 0` is used, so 0 for padding is correct.

        # 3. Transformer Layers
        output = x
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        # 4. Final Normalization
        output = self.norm(output)

        # 5. Output Head
        logits = self.output_head(output) # Shape: (batch_size, seq_len, vocab_size)

        return logits
