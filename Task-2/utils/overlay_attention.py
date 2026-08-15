import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def overlay_attention(attention_matrix, image_path):

    avg_attention=attention_matrix[0, :, 0, 1:].mean(dim=0)

    grid_size=int(math.sqrt(avg_attention.shape[0]))
    attention_2D=avg_attention.reshape(1,1,grid_size,grid_size)

    attention_2D_upsampled=F.interpolate(attention_2D,
                                size=(224,224),
                                mode='bilinear',
                                align_corners=False)
    heatmap = attention_2D_upsampled.squeeze().detach().cpu().numpy()

    heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())


    original_image = Image.open(image_path).resize((224, 224))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5) 
    plt.axis('off')
    plt.show()

