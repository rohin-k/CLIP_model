import torch
import torch.nn as nn

from custom_transformer_block import CustomTransformerBlock

from config import TextEncoderConfig

_default = TextEncoderConfig()

class TextEncoder(nn.Module):
    def __init__(
            self,
            d_model:        int = _default.d_model,
            max_seq_len:    int = _default.max_seq_len,
            vocab_size:     int = _default.vocab_size,
            n_layers:       int = _default.n_layers,
            n_heads:        int = _default.n_heads,
            d_mlp:          int = _default.d_mlp,
            d_embedding:    int = _default.d_embedding,
            pad_token_id:   int = _default.pad_token_id,
            eos_token_id:   int = _default.eos_token_id
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.d_embedding = d_embedding
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.config = {k:v for k, v in locals().items() if k != 'self'}

        # Token embedding.
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=self.pad_token_id
            )
        
        # Positional Embedding
        self.positional_embedding = nn.Parameter(
            nn.init.normal_(torch.empty(1,self.max_seq_len,self.d_model), std=0.01)
            )
        
        # Additive causal mask
        causal_mask = torch.triu(
            torch.full((self.max_seq_len, self.max_seq_len), float("-inf")),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)
        
        # self.n_layers instances of CustomTransformerBlock
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

        # Projection to common embedding space
        self.projection = nn.Parameter(
            nn.init.normal_(torch.empty(self.d_model, self.d_embedding), std=self.d_model ** -0.5)
        )

    
    def forward(
            self,
            x: torch.Tensor,
            padding_mask: torch.Tensor=None
    ) -> torch.Tensor:
        batch, n_tokens = x.shape

        if padding_mask is None:
            padding_mask = (x == self.pad_token_id)[:, None, None, :]
        

        x = self.token_embedding(x)

        x = x + self.positional_embedding[:n_tokens, :]

        causal_mask = self.causal_mask[:n_tokens, :n_tokens]

        for layer in self.layers:
            x = layer(
                x,
                padding_mask = padding_mask,
                causal_mask = causal_mask
            )

        x = self.post_layer_norm(x)

        mask_2d = padding_mask.view(batch, n_tokens)

        # Getting the EOS token indices for each sample in the batch
        eos_indices = (
            (~mask_2d) # only non-padding tokens are True
            .long()         # bool to int, so padding tokens are zeros and the others, ones
            .cumsum(dim=-1)  # cumulative sum, the last non-padding position will have max value
            .argmax(dim=-1)  # returns index of first occurence of max cumulative sum, i.e. EOS token
        )

        # EOS pooling.
        x = x[torch.arange(batch, device=x.device), eos_indices, :]

        # Applying projection
        x = x @ self.projection

        return x



