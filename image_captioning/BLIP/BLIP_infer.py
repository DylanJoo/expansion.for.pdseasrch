from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("/tmp2/chiuws/fine_tuned_BLIP/BLIP-base_pds/checkpoint-16000", torch_dtype=torch.float16)

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

def BLIP_captioning(image_list: list, max_length=10, top_k=10, return_sequences=10, do_sample=True) -> list:
    # Load all images into a list
    images = load_images_concurrently(image_list)

    # Process the images to generate the batched tensors for input
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}

    # Generate the descriptions
    generated_ids = model.generate(**inputs, max_new_tokens=max_length, top_k=top_k, num_return_sequences=return_sequences, do_sample=do_sample)

    # Decode the generated descriptions
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_captions
