import torch
from models_mlsr_wgen import BlipForQuestionAnswering
import requests
from PIL import Image
from transformers import AutoProcessor

model = BlipForQuestionAnswering.from_pretrained(
        "/tmp2/jhju/expansion.for.pdseasrch/models/blip-base-ft-mlsr-plus/checkpoint-20000"
)
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "an image of a cat"

inputs = processor(images=image, text=text, return_tensors="pt")
B, L = inputs['input_ids'].size(0), 10
inputs['decoder_input_ids'] = torch.arange(-1, L+2)[:L].repeat((B, 1))
inputs['decoder_input_ids'][:, 0] = model.decoder_start_token_id

outputs = model(**inputs)

print(outputs.keys())
print(torch.topk(outputs['document_feat'], k=5, dim=-1).indices)
print(torch.topk(outputs['product_feat'], k=5, dim=-1).indices)
print(processor.tokenizer.batch_decode(torch.topk(outputs['document_feat'], k=5, dim=-1).indices))
print(processor.tokenizer.batch_decode(torch.topk(outputs['product_feat'], k=5, dim=-1).indices))
