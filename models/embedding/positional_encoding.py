from torch import *

class PositionalEncoding(nn.Module):
    """
    This module computes sinusoid encoding.

    Why do we need this?

    RNNs have an "intrinsic" way of keeping tabs on the sequential ordering. But in transformers,
    this information is "lost" as we do parallel computations. Thus, we need a way to encode
    the original ordering of the sequence.

    """

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
