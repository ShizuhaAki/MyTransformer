from torch import *

class LayerNormalization(nn.Module):
    """
    Implements Layer Normalization.

    Layer Normalization is used here, instead of Batch normalization
    This is because  if using batch-norm, it would be uncertain what would
    be the appropriate normalization constant (the total number of elements to divide
    by during normalization)  to use. Different batches would have different normalization
    constants which leads to instability during the course of training.

    """
    def __init__(self, d_model, eps=1e-12):
        super(LayerNormalization, self).__init__()
        # nn.Parameter-s are learnable
        self.gamma = nn.Parameter(ones(d_model))
        self.beta = nn.Parameter(zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # Normalization
        output = (x - mean) / sqrt(var + self.eps)
        output = self.gamma * output + self.beta
        return output
