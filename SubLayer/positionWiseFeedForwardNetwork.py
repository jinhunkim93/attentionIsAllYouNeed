import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: model dimension
        d_ff: feed-forward dimension
        dropout: dropout rate
        Input: x of shape (batch_size, seq_len, d_model)
        Output: transformed x of shape (batch_size, seq_len, d_model)
        Note: This implementation applies the feed-forward network to each position separately and identically
        as described in the "Attention is All You Need" paper.
        The feed-forward network consists of two linear transformations with a ReLU activation in between.
        The first linear layer expands the dimension from d_model to d_ff, and the second linear layer projects it back to d_model.
        Dropout is applied after the ReLU activation to prevent overfitting.
        """

        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2 = ReLU(xW1 + b1)W2 + b2 = ReLU(Linear1(x))W2 + b2 = Linear2(ReLU(Linear1(x)))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x