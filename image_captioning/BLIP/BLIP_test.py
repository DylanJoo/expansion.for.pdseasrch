from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("/tmp2/chiuws/fine_tuned_BLIP/BLIP-base_pds/checkpoint-8000", torch_dtype=torch.float16)

model.to(DEVICE)

image_list = [
    '/tmp2/chiuws/testing_images/1.jpg',
    '/tmp2/chiuws/testing_images/2.jpg',
    '/tmp2/chiuws/testing_images/4.jpg',
    '/tmp2/chiuws/testing_images/6.jpg',
]

# 1. Load all images and preprocess them
images = [Image.open(img_path).convert('RGB') for img_path in image_list]
images = [img.resize((224, 224)) for img in images]
inputs = processor(images=images, return_tensors="pt")

# 2. Move them to GPU
inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}

# 3. Generate the descriptions
generated_ids = model.generate(**inputs, max_new_tokens=20, top_k=10, num_return_sequences=10, do_sample=True)

# 4. Decode the generated descriptions
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
