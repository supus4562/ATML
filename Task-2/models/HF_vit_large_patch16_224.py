from transformers import ViTForImageClassification, ViTImageProcessor

def build_vit_HF():
    model_name = 'google/vit-large-patch16-224'
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, attn_implementation='eager')
    return model, processor
