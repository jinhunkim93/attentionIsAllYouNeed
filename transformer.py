# Transformer consisting of Encoder Decoder architecture with multi-head attention and position-wise feed-forward networks
# Includes masking for handling variable-length sequences and padding for sequence-to-sequence tasks
# Input: source sequence and target sequence
# Output: transformed target sequence

import torch
import torch.nn as nn
from Encoder.encoder import Encoder
from Decoder.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, model_dim, num_heads, num_layers)
        self.decoder = Decoder(output_dim, model_dim, num_heads, num_layers)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        source_mask = None
        target_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        enc_output = self.encoder(src, source_mask)
        dec_output = self.decoder(tgt, enc_output, target_mask)
        # apply final linear layer and softmax to get output probabilities
        self.fc_out = nn.Linear(dec_output.size(-1), tgt.size(-1))
        output = self.fc_out(dec_output)
        return output