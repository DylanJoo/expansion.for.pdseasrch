import torch

from models_mlsr_retrieval import BlipForRetrieval
import requests
from PIL import Image
from transformers import AutoProcessor

model = BlipForRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
)
processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-itm-base-coco"
)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "an image of a cat"

inputs = processor(images=image, text=text, return_tensors="pt")
query = "Is this a cat?"
qinputs = processor.tokenizer(query, return_tensors="pt")
outputs = model(**inputs, q_input_ids=qinputs.input_ids, q_attention_mask=qinputs.attention_mask)

# values, _ = torch.max(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
print(torch.topk(outputs['image_features'], k=5, dim=-1).indices)
print(torch.topk(outputs['text_features'], k=5, dim=-1).indices)
print(torch.topk(outputs['multimodal_features'], k=5, dim=-1).indices)
