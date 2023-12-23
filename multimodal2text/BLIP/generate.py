import os
import torch
from tqdm import tqdm 
import json
import argparse
from PIL import Image
from transformers import GenerationConfig, AutoConfig
from datasets import Dataset
from tools import batch_iterator, load_images

def inference(model, processor, batch, config, device, template, **kwargs):
    if kwargs.get('text_only', False):
        blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
        images = [blank] * len(batch['title'])
    else:
        images = [Image.open(img).convert('RGB').resize((384, 384))\
                for img in batch['image']]

    if template:
        texts = [template.format(t, s) \
                for t, s in zip(batch['title'], batch['description'])]
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
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--model_name", default='t5-base', type=str)
    parser.add_argument("--model_hf_name", default='t5-base', type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
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
    from transformers import BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(args.model_name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_hf_name)

    model = model.to(args.device)
    model = model.eval()

    # load config
    generation_config = GenerationConfig(
            num_beams=args.num_beams, 
            top_k=args.top_k, 
            top_p=args.top_p, 
            do_sample=args.do_sample, 
            max_new_tokens=args.max_tgt_length, 
            num_return_sequences=args.num_return_sequences,
    )

    fout = open(args.output_jsonl, 'w')

    # load data: text
    images = load_images(args.img_collection)

    data_both = []
    data_onlytext = []
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data = {'doc_id': item['doc_id'], 
                    'title': item['title'], 
                    'description': item['description']}
            image = images.get(str(item['doc_id']), None)

            if image:
                data.update({'image': image})
                data_both.append(data)
            else:
                data_onlytext.append(data)

    dataset_onlytext = Dataset.from_list(data_onlytext)
    dataset_both = Dataset.from_list(data_both)
    print(dataset_onlytext)
    print(dataset_both)

    # inference: text only
    data_iterator = batch_iterator(dataset_onlytext, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset_onlytext)//args.batch_size+1):
        batch_ids = batch['doc_id']
        batch_titles = batch['title']

        if args.do_text_only:
            # enumerate the generated outputs
            generated_texts = inference(
                    model, processor, batch,
                    config=generation_config,
                    device=args.device,
                    max_src_length=args.max_src_length,
                    template=args.template_src,
                    text_only=True # only the text
            )
            for i in range(len(batch_ids)):
                docid = batch_ids[i]
                title = batch_titles[i]
                if args.num_return_sequences > 1:
                    start = i * args.num_return_sequences
                    end = start+args.num_return_sequences
                    generated_text = ". ".join(generated_texts[start: end])
                else:
                    generated_text = generated_texts[i]

                fout.write(json.dumps({
                    "id": str(docid), 
                    "contents": title + " . " + generated_text,
                })+'\n')
        else:
            for i in range(len(batch_ids)):
                docid = batch_ids[i]
                title = batch_titles[i]
                fout.write(json.dumps({
                    "id": str(docid), 
                    "contents": title
                })+'\n')

    # inference: text + image
    data_iterator = batch_iterator(dataset_both, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset_both)//args.batch_size+1):

        batch_ids = batch['doc_id']
        batch_titles = batch['title']

        # maybe we need the separate (1) both (2) image-only (3) text-only
        generated_texts = inference(
                model, processor, batch,
                config=generation_config,
                device=args.device,
                max_src_length=args.max_src_length,
                template=args.template_src
        )

        # enumerate the generated outputs
        for i in range(len(batch_ids)):
            docid = batch_ids[i]
            title = batch_titles[i]
            if args.num_return_sequences > 1:
                start = i * args.num_return_sequences
                end = start+args.num_return_sequences
                generated_text = ". ".join(generated_texts[start: end])
            else:
                generated_text = generated_texts[i]

            fout.write(json.dumps({
                "id": str(docid), 
                "contents": title + " . " + generated_text,
            })+'\n')

    print("Done")
