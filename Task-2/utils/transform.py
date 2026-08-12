from torchvision import transforms


ImageNet_mean=[0.485, 0.456, 0.406]
ImageNet_std=[0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(ImageNet_mean, ImageNet_std)
])
