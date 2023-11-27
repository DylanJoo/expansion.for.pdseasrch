from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

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

# 2. Create a batched tensor of pixel values
pixel_values_batch = inputs.pixel_values.to(DEVICE)

# 3. Generate the captions in a single forward pass
generated_ids = model.generate(pixel_values=pixel_values_batch, max_new_tokens=20, top_k=10, num_return_sequences=10, do_sample=True)


# 4. Decode the generated captions
generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

for caption in generated_captions:
    print(caption)
    print("=================================================")
