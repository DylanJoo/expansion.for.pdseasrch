from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("/tmp2/chiuws/fine_tuned_BLIP_2/BLIP-base_pds/checkpoint-10000", torch_dtype=torch.float16)

model.to(DEVICE)

image_list = [
    '/tmp2/chiuws/bag.png', 
    '/tmp2/chiuws/bra_top.png', 
    '/tmp2/chiuws/card.jpg',
    '/tmp2/chiuws/gadget_bag.jpg', 
    '/tmp2/chiuws/moon_tshirt.png', 
    '/tmp2/chiuws/starwars_doll.png',
    '/tmp2/chiuws/weird_doll.jpg',
    '/tmp2/chiuws/women_clothes.png',
    '/tmp2/chiuws/phone_case.jpg',
    '/tmp2/chiuws/naruto_doll.jpg',
    '/tmp2/chiuws/cd.png',
    '/tmp2/chiuws/bird_decoration.png'
]

# 1. Load all images and preprocess them
images = [Image.open(img_path).convert('RGB') for img_path in image_list]
inputs = processor(images=images, return_tensors="pt")

# 2. Move them to GPU
inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}

# 3. Generate the descriptions
generated_ids = model.generate(**inputs, max_new_tokens=20)

# 4. Decode the generated descriptions
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

result = []

for generated_text in generated_texts:
    result.append(generated_text.strip())

for caption in result:
    print(caption)
    print("=================================================")
