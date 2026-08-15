import numpy as np
import torch
from PIL import Image

PATCH_SIZE = 16


def mask_patches_rand(image, mask_ratio):
    image = np.array(image.convert("RGB"))  
    H, W, C = image.shape

    num_patches_h = H // PATCH_SIZE
    num_patches_w = W // PATCH_SIZE
    num_patches = num_patches_h * num_patches_w

    num_masked = int(mask_ratio * num_patches)
    masked_indices = np.random.choice(num_patches, num_masked, replace=False)

    masked = image.copy()
    for idx in masked_indices:
        row = (idx // num_patches_w) * PATCH_SIZE
        col = (idx % num_patches_w) * PATCH_SIZE
        masked[row:row+PATCH_SIZE, col:col+PATCH_SIZE, :] = 0

    return Image.fromarray(masked)


def mask_patches_center(image, mask_ratio):
    image = np.array(image.convert("RGB")) 
    H, W, C = image.shape

    num_patches_h = H // PATCH_SIZE
    num_patches_w = W // PATCH_SIZE
    num_patches = num_patches_h * num_patches_w
    num_masked = int(mask_ratio * num_patches)

    center_h, center_w = num_patches_h / 2, num_patches_w / 2
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            dist = (i - center_h) ** 2 + (j - center_w) ** 2
            patches.append((dist, i, j))
    
    patches.sort(key=lambda x: x[0]) 
    patches_to_mask = patches[:num_masked]

    masked = image.copy()
    for _, i, j in patches_to_mask:
        row, col = i * PATCH_SIZE, j * PATCH_SIZE
        masked[row:row+PATCH_SIZE, col:col+PATCH_SIZE, :] = 0

    return Image.fromarray(masked)


def mask_patches_edges(image, mask_ratio):
    image = np.array(image.convert("RGB"))  
    H, W, C = image.shape

    num_patches_h = H // PATCH_SIZE
    num_patches_w = W // PATCH_SIZE
    num_patches = num_patches_h * num_patches_w
    num_masked = int(mask_ratio * num_patches)

    center_h, center_w = num_patches_h / 2, num_patches_w / 2
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            dist = (i - center_h) ** 2 + (j - center_w) ** 2
            patches.append((dist, i, j))

    patches.sort(key=lambda x: x[0], reverse=True) 
    patches_to_mask = patches[:num_masked]

    masked = image.copy()
    for _, i, j in patches_to_mask:
        row, col = i * PATCH_SIZE, j * PATCH_SIZE
        masked[row:row+PATCH_SIZE, col:col+PATCH_SIZE, :] = 0

    return Image.fromarray(masked)