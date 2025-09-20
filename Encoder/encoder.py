# Encoder Module that consists of a stack of N Encoder layers
# Each layer has 2 sub-layers: self-attention and feed-forward network
# Includes residual connections and layer normalization
# Input: source sequence
# Output: transformed source sequence

import torch.nn as nn
from Encoder.encoderLayer import EncoderLayer

class Encoder(nn.Module):
    # d_model: model dimension, num_heads: number of attention heads, d_ff: feed-forward dimension, num_layers: number of encoder layers, dropout: dropout rate
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        # Note: This implementation assumes that the input is already embedded and has positional encoding applied
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    # input : src_seq: source sequence, src_mask: source mask
    # output: transformed source sequence
    def forward(self, src_seq, src_mask=None):
        # Pass the input through each encoder layer in sequence where the output of one layer is the input to the next
        # src_seq -> layer(src_seq, src_mask) -> layer(src_seq, src_mask) -> ... -> layer(src_seq, src_mask) -> src_seq num_layers times
        for layer in self.layers:
            src_seq = layer(src_seq, src_mask)
        return src_seq