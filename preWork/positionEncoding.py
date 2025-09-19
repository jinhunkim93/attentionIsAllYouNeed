# Positional Encoding module
# Implements sinusoidal positional encoding as described in "Attention Is All You Need"
# Input: sequence of vectors (batch_size, seq_len, d_model)
# Output: sequence of vectors with positional encoding added
# PE(pos,2i) = sin(pos/10000^(2i/dmodel))
# PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        # Calculate the positional encodings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        # Register as a buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        # x: (batch_size, seq_len, d_model)
        # pe: (1, max_len, d_model)
        # Ensure the positional encoding is on the same device as the input
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x