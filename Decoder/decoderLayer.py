# Decoder Layer that consists of 3 sub-layers: self-attention, encoder-decoder attention, and feed-forward network wtih residual connections and layer normalization
# Each sub-layer is followed by a layer normalization step
# The output of each sub-layer is added to its input (residual connection) before being normalized
# Input: target sequence and encoder output
# Output: transformed target sequence

import torch.nn as nn
from SubLayer.multiHeadAttention import MultiHeadAttention
from SubLayer.positionWiseFeedForwardNetwork import PositionWiseFeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Three sub-layers: self-attention, encoder-decoder attention, and feed-forward network
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff, dropout)
        # Layer normalization for each sub-layer, implemented with nn.LayerNorm, which is an implementation of "Layer Normalization" paper
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)


    def forward(self, trg_seq, enc_output, src_mask=None, trg_mask=None):
        """
        Decoder Layer forward pass
        inputs: trg_seq: target sequence, enc_output: encoder output, src_mask: source mask, trg_mask: target mask
        output: transformed target sequence
        """

        # Sub-layer 1 : Self-attention
        # Apply self-attention, add residual connection, and normalize
        # Apply target mask to prevent attention to future tokens in the target sequence
        trg_seq = self.norm1(trg_seq + self.dropout(self.self_attn(trg_seq, trg_seq, trg_seq, trg_mask)))

        # Sub-layer 2 : Encoder-decoder attention
        # Apply encoder-decoder attention, add residual connection, and normalize
        # Use the encoder output as key and value, and the target sequence as query
        # Apply source mask to prevent attention to padding tokens in the source sequence
        trg_seq = self.norm2(trg_seq + self.dropout(self.enc_dec_attn(trg_seq, enc_output, enc_output, src_mask)))
        
        # Sub-layer 3 : Position-wise Fully Connected Feed-forward network
        # Apply feed-forward network, add residual connection, and normalize
        # Apply dropout to the output of the feed-forward network before adding the residual connection
        trg_seq = self.norm3(trg_seq + self.dropout(self.ffn(trg_seq)))

        return trg_seq
