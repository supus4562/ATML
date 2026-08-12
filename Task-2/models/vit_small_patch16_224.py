import timm

def build_vit():
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    return model