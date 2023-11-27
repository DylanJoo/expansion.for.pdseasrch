from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

model.to(DEVICE)

def load_images_concurrently(image_list: list) -> list:
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(lambda img_path: Image.open(img_path).convert('RGB'), image_list))
    return images

def BLIP2_captioning(image_list: list) -> list:
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
        