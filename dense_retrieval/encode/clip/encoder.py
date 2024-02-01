import torch
import open_clip

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

class ClipEncoder:
    def __init__(self, model_name: str='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(DEVICE)
        
    def text_encode(self, sentences_list: list) -> torch.Tensor:
        sentences = self.tokenizer(sentences_list).to(DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(sentences)

        return text_features

    def vision_encode(self, images_list: list) -> torch.Tensor:
        # Apply preprocessing to each image individually
        preprocessed_images = [self.preprocess(img).unsqueeze(0) for img in images_list]

        # Concatenate all preprocessed images into a single tensor
        images_tensor = torch.cat(preprocessed_images, dim=0).to(DEVICE)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(images_tensor)

        return image_features