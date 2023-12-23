import os
import torch
from tqdm import tqdm 
import json
import argparse
from PIL import Image
from transformers import GenerationConfig, AutoConfig
from datasets import Dataset
from tools import batch_iterator, load_images, load_collection

def inference(model, processor, batch, config, device, template, **kwargs):
    try:
        images = [Image.open(batch['image']).convert('RGB').resize((384, 384))]
    except:
        images = [Image.new('RGB', (384, 384), color=(255, 255, 255))]

    if template:
        texts = [template.format(batch['title'], batch['description'])]
    else:
        texts = None

    processed_input = processor(
        images=images, text=texts,
        max_length=kwargs.pop('max_src_length', 512),
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)

    outputs = model.generate(**processed_input, generation_config=config)
    processed_output = processor.batch_decode(
            outputs, skip_special_tokens=True
    )
    return processed_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--img_collection", type=str)
    parser.add_argument("--model_name", default='t5-base', type=str)
    parser.add_argument("--model_hf_name", default='t5-base', type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_src_length", default=256, type=int)
    parser.add_argument("--max_tgt_length", default=32, type=int)
    parser.add_argument("--num_return_sequences", default=3, type=int)
    parser.add_argument("--template_src", default=None, type=str)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--do_text_only", action='store_true', default=False)
    args = parser.parse_args()

    # load hf 
    from models_vis_enhanced import BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(args.model_name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_hf_name)
    model = model.to(args.device)
    model = model.eval()

    # load config
    generation_config = GenerationConfig(
            num_beams=args.num_beams, 
            top_k=args.top_k, 
            do_sample=args.do_sample, 
            max_new_tokens=args.max_tgt_length, 
            num_return_sequences=args.num_return_sequences,
    )

    # load data: text
    images = load_images(args.img_collection)
    corpus = load_collection(args.collection, append=False)

    # enumerate the generated outputs
    for docid in [414984, 104175, 640271, 1617168, 467983, 1227643, 910117, 1281957, 832006, 37679, 333358, 1052725, 160049]:
    # for docid in [13]:
        docid = str(docid)
        batch = {
                "title": corpus[docid]['title'],
                "description": corpus[docid]['description'],
                'image': images[docid]
        }
        generated_texts = inference(
                model, processor, batch,
                config=generation_config,
                device=args.device,
                max_src_length=args.max_src_length,
                template=args.template_src,
                text_only=False
        )
        print(batch['title'], generated_texts)
