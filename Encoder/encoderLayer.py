# Encoder Layer that consists of 2 sub-layers: self-attention and feed-forward network with residual connections and layer normalization
# Each sub-layer is followed by a layer normalization step
# The output of each sub-layer is added to its input (residual connection) before being normalized
# Input: source sequence
# Output: transformed source sequence

import torch.nn as nn
from SubLayer.multiHeadAttention import MultiHeadAttention
from SubLayer.positionWiseFeedForwardNetwork import PositionWiseFeedForwardNetwork

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # Two sub-layers: self-attention and feed-forward network

        # Sub-layer 1 : Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Sub-layer 2 : Position-wise Fully Connected Feed-forward network
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff, dropout)

        # Layer normalization for each sub-layer, implemented with nn.LayerNorm, which is an implementation of "Layer Normalization" paper
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    # input : src_seq: source sequence, src_mask: source mask
    # output: transformed source sequence
    def forward(self, src_seq, src_mask=None):
        # Encoder Layer forward pass

        # Sub-layer 1 : Self-Attention
        # Apply self-attention, add residual connection, and normalize
        # Apply source mask to prevent attention to padding tokens in the source sequence
        src_seq = self.norm1(src_seq + self.dropout(self.self_attn(src_seq, src_seq, src_seq, src_mask)))

        # Sub-layer 2 : Position-wise Fully Connected Feed-forward network
        # Apply feed-forward network, add residual connection, and normalize
        # Apply dropout to the output of the feed-forward network before adding the residual connection
        src_seq = self.norm2(src_seq + self.dropout(self.ffn(src_seq)))

        return src_seq