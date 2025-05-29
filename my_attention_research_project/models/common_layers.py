import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements Positional Encoding.

    Args:        d_model (int): The dimension of the model.
        max_len (int, optional): The maximum sequence length. Defaults to 5000.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout_rate: float = 0.1):
        super().__init__()
        self.max_len = max_len  # Store max_len as instance attribute
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)

        # PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape pe to (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        # Register pe as a buffer (not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PositionalEncoding.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
                          with positional encodings added.
        """
        # Add positional encodings to the input embeddings
        # self.pe is (1, max_len, d_model). We take up to seq_len.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward layer.

    Args:
        d_model (int): The dimension of the input and output.
        d_ff (int): The inner dimension of the feed-forward layer.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PositionwiseFeedForward.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pass through linear1, ReLU, dropout, then linear2
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TokenEmbedding(nn.Module):
    """
    Token Embedding layer.
    Simply wraps nn.Embedding.
    Includes an optional scaling factor for the embeddings as used in some Transformer variants.
    """
    def __init__(self, vocab_size: int, d_model: int, scale_grad_by_freq: bool = False, scale_factor: float = None):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embeddings.
            scale_grad_by_freq (bool): If True, gradients w.r.t. input tokens will be scaled by the inverse
                                       of the frequency of the token in the batch. Default is False.
                                       (Note: nn.Embedding's scale_grad_by_freq is about word frequency in general,
                                       not batch. This is a standard nn.Embedding parameter.)
            scale_factor (float, optional): If provided, the embedding outputs are multiplied by this factor.
                                            Some models multiply embeddings by sqrt(d_model). Defaults to None.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, scale_grad_by_freq=scale_grad_by_freq)
        self.d_model = d_model
        self.scale_factor = scale_factor
        if self.scale_factor is None and d_model is not None: # Common practice is to scale by sqrt(d_model)
             # self.scale_factor = math.sqrt(d_model) # Enable if scaling by sqrt(d_model) is desired by default
             pass


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): Input tokens, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens, shape (batch_size, seq_len, d_model).
        """
        embeddings = self.embedding(tokens)
        if self.scale_factor is not None:
            embeddings = embeddings * self.scale_factor
        return embeddings
