from torch import *

class PositionwiseFeedForward(nn.Module):
    """
    This implements the Position-wise feed forward layer described in the original paper

    FFN(x) = ReLU(x * W1 + b1) * W2+b2
    """
    def __init__(self, d_model: int, d_hidden: int, drop_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x