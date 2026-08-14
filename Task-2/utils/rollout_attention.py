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
    # Extract sequence length from the first layer (e.g., 197)
    seq_len = attentions[0].shape[-1]
    
    # Initialize R (Aggregation Matrix) as an Identity Matrix
    R = torch.eye(seq_len).to(attentions[0].device)
    
    for layer_attention in attentions:
        # 1. Mean Matrix Across Heads (for the first image in batch)
        A = layer_attention[0].mean(dim=0)
        
        # 2. Add identity
        A_hat = A + torch.eye(seq_len).to(A.device)
        
        # 3. Normalization (row-wise so probabilities sum to 1)
        A_tilde = A_hat / A_hat.sum(dim=-1, keepdim=True)
        
        # 4. Aggregation (Matrix multiplication of current layer with rolling total)
        R = torch.matmul(A_tilde, R)
        
    # Extract the [CLS] token's cumulative attention to all image patches (skip index 0)
    cls_attention = R[0, 1:]
    
    return cls_attention

def overlay_rollout(attentions, image_path):
    """
    Upsamples the rollout matrix and overlays it on the original image.
    """
    # Get the 1D aggregated attention array
    cls_attention = compute_attention_rollout(attentions)
    
    # Calculate grid size (14x14) and reshape
    grid_size = int(math.sqrt(cls_attention.shape[0]))
    attention_2d = cls_attention.reshape(1, 1, grid_size, grid_size)
    
    # Upsample to 224x224 using bilinear interpolation
    upsampled = F.interpolate(attention_2d, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = upsampled.squeeze().detach().cpu().numpy()
    
    # 5. Final Normalization (Min-Max scaling for the heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Overlay and Plot
    original_image = Image.open(image_path).resize((224, 224))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()