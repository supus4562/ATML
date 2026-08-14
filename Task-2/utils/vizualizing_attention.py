import matplotlib.pyplot as plt
import math

NUM_HEADS = 14
NUM_IMAGES_IN_BATCH = 1

def visualize_attention(attention_matrix):

    figs, axs = plt.subplots(4, 4, figsize=(12, 8), layout='constrained')
    axs = axs.flatten()
    
    for j in range(NUM_IMAGES_IN_BATCH):
        for i in range(NUM_HEADS):
            plot_idx = j * NUM_HEADS + i
            attention_1d = attention_matrix[j, i, 0, 1:].detach().cpu().numpy()
            grid_size = int(math.sqrt(attention_1d.shape[0]))
            attention_2d = attention_1d.reshape((grid_size, grid_size))
            axs[plot_idx].imshow(attention_2d)
            axs[plot_idx].axis('off')