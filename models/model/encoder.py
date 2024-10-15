from torch import *
from models.layers.attention import MultiHeadAttention
from models.layers.layernorm import LayerNormalization
from models.layers.ffn import PositionwiseFeedForward
from models.embedding.embedding import TransformerEmbedding
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, drop_rate)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(p=drop_rate)


    def forward(self, x, mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_hidden, n_heads, n_layers, drop_rate, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_rate, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_heads, drop_rate) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x