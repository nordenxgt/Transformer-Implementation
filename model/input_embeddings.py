import math

from torch import nn, Tensor

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)