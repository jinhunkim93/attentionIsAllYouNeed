# attention.py
# Attention calculation module
# Implements scaled dot-product attention and multi-head attention mechanisms
# Input: query, key, value, and optional mask
# Output: attention output and attention weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attention, self).__init__()
        self.scale = 1 / (d_model ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Calculate scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights