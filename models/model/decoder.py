from models.layers.attention import MultiHeadAttention
from models.layers.layernorm import LayerNormalization
from models.layers.ffn import PositionwiseFeedForward
from models.embedding.embedding import TransformerEmbedding
from torch import *
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_rate):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(drop_rate)

        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(drop_rate)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=ffn_hidden, drop_rate=drop_rate)
        self.norm3 = LayerNormalization(d_model=d_model)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, decoder, encoder, decoder_mask, encoder_mask):
        # Apply self-attention on decoder sequence
        xx = decoder
        x = self.self_attention(q=decoder, k=decoder, v=decoder, mask=decoder_mask)
        x = self.dropout1(x)
        x = self.norm1(x + xx)

        # Apply cross attention on encoder sequence
        if encoder is not None:
            xx = encoder
            x = self.cross_attention(q=x, k=encoder, v=encoder, mask=encoder_mask)
            x = self.dropout2(x)
            x = self.norm2(x + xx)

        # Finally, apply position-wise feed forward on the result
        xx = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + xx)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_vocab_size, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_rate, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=decoder_vocab_size,
            d_model=d_model,
            max_len=max_len,
            drop_rate=drop_rate,
            device=device
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                ffn_hidden=ffn_hidden,
                n_heads=n_heads,
                drop_rate=drop_rate
            ) for _ in range(n_layers)
        ])

        self.linear = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, decoder, encoder, decoder_mask, encoder_mask):
        decoder = self.embedding(decoder)

        for layer in self.layers:
            decoder = layer(decoder, encoder, decoder_mask, encoder_mask)

        output = self.linear(decoder)
        return output
