import torch
import torch.nn as nn
from custom_attention import CustomAttention
from custom_mlp import CustomMLP

class CustomTransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_mlp: int
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attn_layer = CustomAttention(d_model=d_model, n_heads=n_heads)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.mlp_layer = CustomMLP(d_model=d_model, d_mlp=d_mlp)

    def forward(
            self,
            x: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            causal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        # Attention layer
        x_normalized_1 = self.layer_norm_1(x)
        attn_out = self.attn_layer(
            x_normalized_1,
            padding_mask=padding_mask,
            causal_mask=causal_mask
            )
        x = x + attn_out # Residual connection

        # MLP layer
        x_normalized_2 = self.layer_norm_2(x)
        mlp_out = self.mlp_layer(x_normalized_2)
        x = x + mlp_out # Residual connection

        return x