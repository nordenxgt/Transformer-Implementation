import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        pos =  torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        freq = torch.pow(10_000, -torch.arange(0, d_model, step=2, dtype=torch.float) / d_model)

        self.pe = torch.zeros(seq_len, d_model).requires_grad(False)
        self.pe[:, 0::2] = torch.sin(pos / freq)
        self.pe[:, 1::2] = torch.cos(pos / freq)

        self.register_buffer("positional_encodings", self.pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_embeddings: Tensor) -> Tensor:
        return self.dropout(input_embeddings + self.pe[:, :input_embeddings.shape[1], :])