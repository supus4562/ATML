import json


def get_label(top1_indices):
    with open("imagenet-simple-labels.json") as f:
        labels = json.load(f)

    for i, idx in enumerate(top1_indices):
        print(f"Image {i}: {labels[idx.item()]}")