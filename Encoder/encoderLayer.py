# Encoder Layer that consists of 2 sub-layers: self-attention and feed-forward network with residual connections and layer normalization
# Each sub-layer is followed by a layer normalization step
# The output of each sub-layer is added to its input (residual connection) before being normalized
# Input: source sequence
# Output: transformed source sequence

import torch.nn as nn
from SubLayer.multiHeadAttention import MultiHeadAttention
from SubLayer.positionWiseFeedForward import PositionWiseFeedForward
# from SubLayer.layerNormalization import LayerNormalization

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        # self.norm1 = LayerNormalization(d_model)
        # self.norm2 = LayerNormalization(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention sub-layer
        # Add residual connection and normalize
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))

        # Position-wise Fully Connected Feed-forward network sub-layer
        # Add residual connection and normalize
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x