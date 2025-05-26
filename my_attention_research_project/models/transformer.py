import torch
import torch.nn as nn
import copy
# from .attention.vanilla_mha import MultiHeadAttention # Example for type hinting
# from .common_layers import PositionwiseFeedForward # Example for type hinting

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.

    It consists of a self-attention mechanism and a position-wise feed-forward network.
    Layer normalization and residual connections are applied around each of these sub-layers.

    Args:
        d_model (int): The number of expected features in the input (required).
        attention_module (nn.Module): An instance of an attention mechanism
                                      (e.g., MultiHeadAttention).
        feed_forward_module (nn.Module): An instance of a feed-forward network
                                         (e.g., PositionwiseFeedForward).
        dropout_rate (float): The dropout value.
    """
    def __init__(self, d_model: int, attention_module: nn.Module, feed_forward_module: nn.Module, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.attention = attention_module
        self.ffn = feed_forward_module

        # Layer normalization for the attention block
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) # Common epsilon value
        # Layer normalization for the feed-forward block
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6) # Common epsilon value

        # Dropout for both sub-layers
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the TransformerEncoderLayer.

        Args:
            src (torch.Tensor): The sequence to the encoder layer.
                                Shape: (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): The mask for the src sequence.
                                               Its shape depends on the attention mechanism.
                                               For self-attention, it might be
                                               (batch_size, 1, 1, seq_len) for padding or
                                               (1, 1, seq_len, seq_len) for causal.
                                               Defaults to None.

        Returns:
            torch.Tensor: The output of the encoder layer.
                          Shape: (batch_size, seq_len, d_model).
        """
        # 1. Attention block
        # Apply layer normalization before self-attention
        src_normed = self.norm1(src)
        # Compute attention output. For encoder self-attention, Q, K, V are all from src_normed.
        # The attention_module is expected to return (output, attention_weights)
        att_output, _ = self.attention(query=src_normed, key=src_normed, value=src_normed, mask=src_mask)
        # Apply dropout to the attention output and add residual connection
        src = src + self.dropout(att_output)

        # 2. Feed-forward block
        # Apply layer normalization before the feed-forward network
        src_normed = self.norm2(src)
        # Compute feed-forward output
        ffn_output = self.ffn(src_normed)
        # Apply dropout to the FFN output and add residual connection
        src = src + self.dropout(ffn_output)

        return src

class TransformerEncoder(nn.Module):
    """
    The Transformer Encoder, composed of a stack of N identical TransformerEncoderLayers.

    Args:
        encoder_layer (nn.Module): An instance of TransformerEncoderLayer.
                                   This layer will be deep-copied N times.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        d_model (int): The dimension of the model, used for the final LayerNorm.
    """
    def __init__(self, encoder_layer: nn.Module, num_layers: int, d_model: int):
        super().__init__()
        # Create a list of N identical encoder layers using deepcopy
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        # Final layer normalization applied after all encoder layers
        self.norm = nn.LayerNorm(d_model, eps=1e-6) # Common epsilon value

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the TransformerEncoder.

        Args:
            src (torch.Tensor): The sequence to the encoder.
                                Shape: (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): The mask for the src sequence,
                                               passed to each encoder layer.
                                               Defaults to None.

        Returns:
            torch.Tensor: The output of the encoder.
                          Shape: (batch_size, seq_len, d_model).
        """
        output = src
        # Pass the input through each encoder layer in sequence
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        # Apply final layer normalization
        if self.norm is not None: # Check if normalization is defined
            output = self.norm(output)

        return output
