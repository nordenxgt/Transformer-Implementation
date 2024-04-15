from torch import nn, Tensor
from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, log_attn: bool = False) -> None:
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.log_attn = log_attn
        self.attn_visual = None
    
    # IMPROVEMENT: use einops for rearraging easily
    def _split(self, tensor: Tensor) -> Tensor:
        return tensor.view(tensor.shape[0], tensor.shape[1], self.h, self.d_k).transpose(1, 2)

    def _concat(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(1, 2).contigous().view(tensor.shape[0], tensor.shape[2], tensor.shape[1]*tensor.shape[3])

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)
        q, k, v = self._split(q), self._split(k), self._split(v)
        out, attn = ScaledDotProductAttention(q, k, v, mask, self.dropout)
        out = self._concat(out)
        if self.log_attn: self.attn_visual = attn
        return self.W_o(out)