from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from keybert import KeyBERT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16)

model.to(DEVICE)

image_list = [
    '/tmp2/chiuws/bag.png', 
    '/tmp2/chiuws/bra_top.png', 
    '/tmp2/chiuws/card.jpg',
    '/tmp2/chiuws/gadget_bag.jpg', 
    '/tmp2/chiuws/moon_tshirt.png', 
    '/tmp2/chiuws/naruto_doll.jpg'
]

# Load all images into a list
images = [Image.open(img_path).convert('RGB') for img_path in image_list]

# Process the images to generate the batched tensors for input
inputs = processor(images, return_tensors="pt")
inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}

# Generate the descriptions
generated_ids = model.generate(**inputs, max_new_tokens=20)

# Decode the generated descriptions
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(type(generated_texts))

key_bert_model = KeyBERT('distilbert-base-nli-mean-tokens')

for generated_text in generated_texts:
    caption = generated_text.strip()
    print(caption)
    keywords = key_bert_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), top_n=3)
    print("key terms:{}".format(keywords))
    print("=================================================")
