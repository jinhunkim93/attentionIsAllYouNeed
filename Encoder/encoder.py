# Encoder Module that consists of a stack of N Encoder layers
# Each layer has 2 sub-layers: self-attention and feed-forward network
# Includes residual connections and layer normalization
# Input: source sequence
# Output: transformed source sequence

import torch.nn as nn
from Encoder.encoderLayer import EncoderLayer

class Encoder(nn.Module):
    # d_model: model dimension, num_heads: number of attention heads, d_ff: feed-forward dimension, num_layers: number of encoder layers, dropout: dropout rate
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        # Create a stack of encoder layers
        # Each layer is an instance of EncoderLayer
        # The layers are stored in a ModuleList to ensure they are registered as sub-modules
        # This allows the parameters of the layers to be learned during training
        # The number of layers is determined by the num_layers parameter
        # Each layer has its own set of parameters
        # The dropout parameter is used to prevent overfitting
        # The d_model, num_heads, and d_ff parameters define the architecture of each layer
        # The input to the encoder is a sequence of vectors, each of dimension d_model
        # The output is a sequence of the same length, with each vector transformed by the encoder layers
        # The src_mask parameter can be used to mask out certain positions in the input sequence
        # This is useful for handling variable-length sequences and padding
        # The forward method processes the input through each layer in sequence
        # The output of one layer is the input to the next
        # Finally, the transformed sequence is returned
        # Example usage:
        # encoder = Encoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6)
        # src = torch.randn(batch_size, seq_length, 512)  # Example input
        # output = encoder(src)  # Transformed output

        # Note: This implementation assumes that the input is already embedded and has positional encoding applied
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    # input : x: source sequence, src_mask: source mask
    # output: transformed source sequence
    def forward(self, x, src_mask=None):
        # Pass the input through each encoder layer in sequence where the output of one layer is the input to the next
        for layer in self.layers:
            x = layer(x, src_mask)
        return x