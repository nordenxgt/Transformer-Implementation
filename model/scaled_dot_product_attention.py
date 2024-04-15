from typing import Tuple

import math
import torch
from torch import nn, Tensor

def ScaledDotProductAttention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, dropout: nn.Module) -> Tuple[Tensor, Tensor]:
    d_k = q.shape[-1]
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: attn.masked_fill_(mask == 0, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    if dropout is not None: attn = dropout(attn)
    return attn @ v, attn