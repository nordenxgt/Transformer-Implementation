from torch import nn, Tensor

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.FFN(x)
