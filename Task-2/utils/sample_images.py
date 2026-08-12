from torchvision import datasets
from torch.utils.data import Subset
from transform import transform
import matplotlib.pyplot as plt


def get_subset():
    dataset = datasets.ImageFolder('data/', transform=transform)
    subset = Subset(dataset, indices=[0, 1, 2])
    return subset


def visualize():
    subset = get_subset()           
    for i, (img, label) in enumerate(subset):
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Image {i}")
        plt.show()