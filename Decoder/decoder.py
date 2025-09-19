# Decoder Module that consists of a stack of N Decoder layers
# Each layer has 3 sub-layers: self-attention, encoder-decoder attention, and feed-forward network
# Includes residual connections and layer normalization
# Input: target sequence and encoder output
# Output: transformed target sequence

import torch.nn as nn
from Decoder.decoderLayer import DecoderLayer

class Decoder(nn.Module):
    # d_model: model dimension, num_heads: number of attention heads, d_ff: feed-forward dimension, num_layers: number of decoder layers, dropout: dropout rate
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    # input : x: target sequence, enc_output: encoder output, src_mask: source mask, tgt_mask: target mask
    # output: transformed target sequence
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Pass the input through each decoder layer in sequence
        for layer in self.layers:
            # Pass the input through the layer with the encoder output and masks where the output of one layer is the input to the next
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x