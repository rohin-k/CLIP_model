import torch
from torch import nn

class CustomMLP(nn.Module):
    """Simple feed-forward block used inside the transformer."""

    def __init__(
            self,
            d_model: int, # dimension of residual stream
            d_mlp: int    # feedforward dimension
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(d_mlp, d_model)

    def forward(
            self,
            x: torch.Tensor # output from attention layer
    ) -> torch.Tensor:
        x_mid = self.up_proj(x)
        x_act = self.act(x_mid)
        out = self.down_proj(x_act)
        return out