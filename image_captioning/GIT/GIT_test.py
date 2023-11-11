from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
from keybert import KeyBERT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on {}...".format(DEVICE))

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

model.to(DEVICE)

image_list = [
    '/tmp2/chiuws/bag.png', 
    '/tmp2/chiuws/bra_top.png', 
    '/tmp2/chiuws/card.jpg',
    '/tmp2/chiuws/gadget_bag.jpg', 
    '/tmp2/chiuws/moon_tshirt.png', 
    '/tmp2/chiuws/naruto_doll.jpg'
]

# 1. Load all images and preprocess them
images = [Image.open(img_path).convert('RGB') for img_path in image_list]
inputs = processor(images=images, return_tensors="pt")

# 2. Create a batched tensor of pixel values
pixel_values_batch = inputs.pixel_values.to(DEVICE)

# 3. Generate the captions in a single forward pass
generated_ids = model.generate(pixel_values=pixel_values_batch, max_length=50)

# 4. Decode the generated captions
generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Load a model (you can also use 'distilbert-base-nli-mean-tokens' or other transformer models)
key_bert_model = KeyBERT('distilbert-base-nli-mean-tokens')

for caption in generated_captions:
    print(caption)
    keywords = key_bert_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), top_n=3)
    print("key terms:{}".format(keywords))
    print("=================================================")
