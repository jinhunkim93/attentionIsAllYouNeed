# Transformer consisting of Encoder Decoder architecture with multi-head attention and position-wise feed-forward networks
# Includes masking for handling variable-length sequences and padding for sequence-to-sequence tasks
# Input: source sequence and target sequence
# Output: transformed target sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder.encoder import Encoder
from Decoder.decoder import Decoder
from PreProcessing.positionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src, trg, d_model, enc_num_heads, enc_num_layers, enc_d_ff, dec_num_heads, dec_num_layers, def_d_ff, dropout=0.1):
        super(Transformer, self).__init__()

        self.src_embed = nn.Embedding(src.size(1), d_model)
        self.src_positionalEncoding = PositionalEncoding(d_model)
        self.trg_embed = nn.Embedding(trg.size(1), d_model)
        self.trg_positionalEncoding = PositionalEncoding(d_model)
        self.embed_weight = torch.sqrt(torch.FloatTensor([d_model]))
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, enc_num_layers, enc_num_heads, enc_d_ff, dropout)
        self.decoder = Decoder(d_model, dec_num_layers, dec_num_heads, def_d_ff, dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg):
        # Step 1 : Pre Processing
        src_pos = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device)
        trg_pos = torch.arange(0, trg.size(1)).unsqueeze(0).repeat(trg.size(0), 1).to(trg.device)
        # embed * weight, add positional encoding, then apply dropout
        src = self.dropout((self.src_embed(src) * self.embed_weight) + self.src_positionalEncoding(src_pos))
        trg = self.dropout((self.trg_embed(trg) * self.embed_weight) + self.trg_positionalEncoding(trg_pos))

        # Step 2 : Encoder-Decoder architecture
        source_mask = None
        target_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        # pass through encoder and decoder
        enc_output = self.encoder(src, source_mask)
        dec_output = self.decoder(trg, enc_output, target_mask)

        # Step 3 : Post Processing
        # apply final linear layer and softmax to get output probabilities
        fc_out = nn.Linear(dec_output.size(-1), trg.size(1)).to(trg.device)
        output = F.softmax(fc_out(dec_output), dim=-1)

        return output