import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def compute_attention_rollout(attentions):
    """
    Computes the Attention Rollout across all layers.
    attentions: Tuple of tensors from model output (one per layer).
    """
    seq_len = attentions[0].shape[-1]
    
    R = torch.eye(seq_len).to(attentions[0].device)
    
    for layer_attention in attentions:
        A = layer_attention[0].mean(dim=0)
        A_hat = A + torch.eye(seq_len).to(A.device)

        A_tilde = A_hat / A_hat.sum(dim=-1, keepdim=True)
        R = torch.matmul(A_tilde, R)
        
    cls_attention = R[0, 1:]
    
    return cls_attention

def overlay_rollout(attentions, image_path):
    """
    Upsamples the rollout matrix and overlays it on the original image.
    """
    cls_attention = compute_attention_rollout(attentions)
    
    grid_size = int(math.sqrt(cls_attention.shape[0]))
    attention_2d = cls_attention.reshape(1, 1, grid_size, grid_size)
    
    upsampled = F.interpolate(attention_2d, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = upsampled.squeeze().detach().cpu().numpy()
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    original_image = Image.open(image_path).resize((224, 224))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()