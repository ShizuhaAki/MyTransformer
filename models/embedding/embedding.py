from torch import *
from models.embedding.positional_encoding import PositionalEncoding


class WordEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_rate, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.word_emb = WordEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        word_emb = self.word_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(word_emb + pos_emb)