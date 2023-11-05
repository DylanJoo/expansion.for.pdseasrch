from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

model.to(DEVICE)

def load_image(img_path: str):
    try:
        return Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Warning: Could not identify image file '{img_path}', it may be damaged.")
        # Creating a black image as a placeholder
        return Image.new('RGB', (224, 224))

def load_images_concurrently(image_list: list) -> list:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_list))
    return images

def GIT_captioning(image_list: list) -> list:
    # 1. Load all images and preprocess them
    images = load_images_concurrently(image_list)
    inputs = processor(images=images, return_tensors="pt")

    # 2. Create a batched tensor of pixel values
    pixel_values_batch = inputs.pixel_values.to(DEVICE)

    # 3. Generate the captions in a single forward pass
    generated_ids = model.generate(pixel_values=pixel_values_batch, max_length=20)

    # 4. Decode the generated captions
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_captions