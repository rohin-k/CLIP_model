import math

import torch
import torch.nn as nn

from vision_encoder import VisionEncoder
from text_encoder import TextEncoder


class CLIP(nn.Module):
    def __init__(
            self,
            config=None # The config for the encoders are set in config.py
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(
            self,
            images: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.vision_encoder(images)

        # L2 normalization
        normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return normalized_embeddings
    
    def encode_text(
            self,
            texts: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.text_encoder(texts)

        # L2 normalization
        normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return normalized_embeddings
    
    def forward(
            self,
            images: torch.Tensor,
            texts: torch.Tensor
    )-> torch.Tensor:

        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        logit_scale = self.logit_scale.exp()

        # cosine similarity matrix of texts w.r.t images
        logits_per_image = image_embeddings @ text_embeddings.t()

        #scaling logits
        logits_per_image = logit_scale * logits_per_image

        # cosine similarity matrix of images w.r.t texts
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text