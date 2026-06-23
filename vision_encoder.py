import torch
import torch.nn as nn

from custom_transformer_block import CustomTransformerBlock

from config import VisionEncoderConfig

_default = VisionEncoderConfig()

class VisionEncoder(nn.Module):
    def __init__(
            self,
            d_model:        int = _default.d_model,
            image_size:     int = _default.image_size,
            patch_size:     int = _default.patch_size,
            n_layers:       int = _default.n_layers,
            n_heads:        int = _default.n_heads,
            d_mlp:          int = _default.d_mlp,
            d_embedding:    int = _default.d_embedding,
            n_patches:      int = _default.n_patches
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.d_embedding = d_embedding
        self.n_patches = n_patches

        self.config = {k:v for k, v in locals().items() if k != 'self'}

        # Patch embedding.
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )

        # CLS token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Positional embedding.
        self.positional_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(1, self.n_patches + 1, self.d_model), std=0.02)
        )

        # Pre layer norm
        self.pre_layer_norm = nn.LayerNorm(self.d_model)

        # Creating self.n_layers instances of
        # CustomTransformerBlock
        self.layers = nn.ModuleList([
            CustomTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_mlp=self.d_mlp # typically 4 times d_model
            )
            for _ in range(self.n_layers)
        ])

        # Post layer norm
        self.post_layer_norm = nn.LayerNorm(self.d_model)

        # Projection to common embedding space.
        self.projection = nn.Linear(self.d_model, self.d_embedding, bias=False)
        nn.init.normal_(self.projection.weight, std = self.d_model ** -0.5)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape

        x = self.patch_embedding(x)

        x = x.flatten(2).transpose(1,2)

        cls_token = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.positional_embedding

        x = self.pre_layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.post_layer_norm(x)

        # CLS pooling.
        x = x[:, 0, :]

        x = self.projection(x)

        return x