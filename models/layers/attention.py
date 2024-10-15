from torch import *
import math


class ScaledDotProductAttention(nn.Module):
    """
    This implements how the attention score is computed, using the
    scale dot attention as proposed in the original paper.

    This class is a subclass of torch.nn.Module, which requires the `forward` method
    to be overridden. The `forward` method should implement the actual computation logic
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        batch_size, num_heads, seq_len, d_tensor = q.size()

        # Compute the attention scoring (alpha)
        k_T = k.transpose(2, 3)

        # Dividing by sqrt(d_tensor) implements the "scaled" in "ScaledDotProductAttention"
        # This is to counteract the trend of dot products becoming too large for larger d_tensors
        # which will push the softmax function to the "flatter" ranges
        score = q @ k_T / math.sqrt(d_tensor)

        # In decoders, we need to mask the computation of attention
        # So that it only attends to previous tokens, this is controlled by the param mask
        if mask is not None:
            # mask is itself a tensor that takes 0 when masking should happen, 1 otherwise
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)

        # Finally, multiply with value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split_heads(self, cur_tensor):
        """This splits the given tensor based on the number of heads in the MHA"""
        batch_size, length, d_model = cur_tensor.size()
        d_tensor = d_model // self.n_heads
        cur_tensor = cur_tensor.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)
        return cur_tensor

    def concat_heads(self, cur_tensor):
        """This is the invert of split_heads"""
        batch_size, head, length, d_tensor = cur_tensor.size()
        d_model = head * d_tensor
        cur_tensor = cur_tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return cur_tensor

    def forward(self, q, k, v, mask=None):
        # Learn the weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Compute the transformed tensors, using the attention mechanism
        output = self.attention(q, k, v, mask)
        output = self.concat_heads(output)
        output = self.w_concat(output)
        return output

