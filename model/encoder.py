from torch import nn, Tensor

from .input_embeddings import InputEmbeddings
from .positional_encoding import PositionalEncoding
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_networks import PositionWiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, ffn_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src: Tensor, src_mask: Tensor):
        x = src
        src = self.mha(q=src, k=src, v=src, mask=src_mask)
        src = self.norm1(x + self.dropout1(src))
        x = src
        src = self.ffn(src)
        src = self.norm2(x + self.dropout2(src))
        return src

class Encoder(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, src_vocab: int, seq_len: int, dropout: float = 0.1, n_layers: int = 6):
        super().__init__()
        self.src_emb = InputEmbeddings(src_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, h, ffn_hidden, dropout) for _ in range(n_layers)])
    
    def forward(self, src: Tensor, src_mask) -> Tensor:
        src = self.dropout(self.pos_enc(src) + self.src_emb(src))
        for layer in self.encoder:
            src = layer(src, src_mask)
        return src