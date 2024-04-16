import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, src, trg) -> None:
        pass