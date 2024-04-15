from torch import nn, Tensor

from .input_embeddings import InputEmbeddings
from .positional_encoding import PositionalEncoding
from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward_networks import PositionWiseFFN

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.mmha = MultiHeadAttention(d_model, h)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, h)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, src: Tensor, trg: Tensor, trg_mask: Tensor, src_mask: Tensor) -> Tensor:
        x = self.mmha(q=trg, k=trg, v=trg, mask=trg_mask)
        x = self.norm1(trg + self.dropout1(x))
        if src is not None:
            _x = x
            x = self.mha(q=x, k=src, v=src, mask=src_mask)
            x = self.norm2(_x + self.dropout2(x))
        _x = x
        x = self.ffn(x)
        x = self.norm3(_x + self.dropout3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, trg_vocab: int, seq_len: int, dropout: float = 0.1, n_layers: int = 6):
        super().__init__()
        self.trg_emb = InputEmbeddings(trg_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, h, ffn_hidden, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:
        trg = self.trg_emb(trg)
        for layer in self.decoder:
            trg = layer(src, trg, src_mask, trg_mask)
        return self.linear(trg)