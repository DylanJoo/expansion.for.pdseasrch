from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)

model.to(DEVICE)

def load_image(img_path: str, error_log_path="damaged_images_BLIP.txt"):
    try:
        return Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Warning: Could not identify image file '{img_path}', it may be damaged.")
        with open(error_log_path, 'a') as error_log:
            error_log.write(f"{img_path}\n")
        # Creating a black image as a placeholder
        return Image.new('RGB', (224, 224))

def load_images_concurrently(image_list: list) -> list:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_list))
    return images

def BLIP_captioning(image_list: list) -> list:
    # Load all images into a list
    images = load_images_concurrently(image_list)

    # Process the images to generate the batched tensors for input
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}

    # Generate the descriptions
    generated_ids = model.generate(**inputs, max_new_tokens=20)

    # Decode the generated descriptions
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    result = []

    for generated_text in generated_texts:
        result.append(generated_text.strip())
        
    return result
        