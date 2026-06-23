import torch
import torch.nn.functional as F

def clip_loss(
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        smoothing: float =0.0
) -> torch.Tensor:
    
    batch = logits_per_image.shape[0]
    labels = torch.arange(batch, device=logits_per_image.device)
    loss_images = F.cross_entropy(logits_per_image, labels, label_smoothing=smoothing)
    loss_texts = F.cross_entropy(logits_per_text, labels, label_smoothing=smoothing)

    return (loss_images + loss_texts) / 2
