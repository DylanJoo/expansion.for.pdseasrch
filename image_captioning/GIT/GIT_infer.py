from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, UnidentifiedImageError
import torch
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

model.to(DEVICE)

def load_image(img_path: str):
    try:
        # Open the image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Resize the image to the target size
        img = img.resize((224, 224))
        return img
    except UnidentifiedImageError:
        print(f"Warning: Could not identify image file '{img_path}', it may be damaged.")
        # Creating a black image as a placeholder
        return Image.new('RGB', (224, 224))

def load_images_concurrently(image_list: list) -> list:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_list))
    return images

def GIT_captioning(image_list: list, max_length=10, top_k=10, return_sequences=10, do_sample=True) -> list:
    # 1. Load all images and preprocess them
    images = load_images_concurrently(image_list)
    inputs = processor(images=images, return_tensors="pt")

    # 2. Create a batched tensor of pixel values
    pixel_values_batch = inputs.pixel_values.to(DEVICE)

    # 3. Generate the captions in a single forward pass
    generated_ids = model.generate(pixel_values=pixel_values_batch, max_new_tokens=max_length, top_k=top_k, num_return_sequences=return_sequences, do_sample=do_sample)

    # 4. Decode the generated captions
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_captions
