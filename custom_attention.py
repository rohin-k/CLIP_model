import math

import torch
from torch import nn


class CustomAttention(nn.Module):
    """Multi-head self-attention for token sequences."""

    def __init__(
            self,
            d_model: int, # dimension of residual stream
            n_heads: int  # number of attention heads
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.o_proj = nn.Linear(d_model, d_model)


    def forward(
            self,
            x:              torch.Tensor,               # batch of patch or token embeddings
            padding_mask:   torch.Tensor | None = None, # batch of masks that are True where padding token is present
            causal_mask:    torch.Tensor | None = None  # batch of masks that avoids bi-directional attention by forcing each token to attend only to the previous ones
    ) -> torch.Tensor:

        batch, num_tokens, d_model = x.shape

        q = self.q_proj(x).view(batch, num_tokens, self.n_heads, self.d_head).transpose(1,2)
        k = self.k_proj(x).view(batch, num_tokens, self.n_heads, self.d_head).transpose(1,2)
        v = self.v_proj(x).view(batch, num_tokens, self.n_heads, self.d_head).transpose(1,2)


        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if padding_mask is not None:
            attn_scores.masked_fill_(padding_mask, float("-inf"))

        if causal_mask is not None:
            attn_scores = attn_scores + causal_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        out = attn_weights @ v

        out = out.transpose(1,2).contiguous().view(batch, num_tokens, d_model)

        return self.o_proj(out)