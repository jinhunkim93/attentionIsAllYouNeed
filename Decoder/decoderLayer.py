# Decoder Layer that consists of 3 sub-layers: self-attention, encoder-decoder attention, and feed-forward network wtih residual connections and layer normalization
# Each sub-layer is followed by a layer normalization step
# The output of each sub-layer is added to its input (residual connection) before being normalized
# Input: target sequence and encoder output
# Output: transformed target sequence

import torch.nn as nn
from SubLayer.multiHeadAttention import MultiHeadAttention
from SubLayer.positionWiseFeedForward import PositionWiseFeedForward
# from SubLayer.layerNormalization import LayerNormalization

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Three sub-layers: self-attention, encoder-decoder attention, and feed-forward network
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        # self.norm1 = LayerNormalization(d_model)
        # self.norm2 = LayerNormalization(d_model)
        # self.norm3 = LayerNormalization(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # input : x: target sequence, enc_output: encoder output, src_mask: source mask, tgt_mask: target mask
    # output: transformed target sequence
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention sub-layer
        # Apply self-attention, add residual connection, and normalize
        # Apply target mask to prevent attention to future tokens in the target sequence
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        
        # Encoder-decoder attention sub-layer
        # Apply encoder-decoder attention, add residual connection, and normalize
        # Use the encoder output as key and value, and the target sequence as query
        # Apply source mask to prevent attention to padding tokens in the source sequence
        x = self.norm2(x + self.dropout(self.enc_dec_attn(x, enc_output, enc_output, src_mask)))
        
        # Feed-forward network sub-layer
        # Apply feed-forward network, add residual connection, and normalize
        # Apply dropout to the output of the feed-forward network before adding the residual connection
        x = self.norm3(x + self.dropout(self.ffn(x)))
    
        return x
