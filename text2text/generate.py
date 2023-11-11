import os
import torch
from tqdm import tqdm 
import json
import argparse
from transformers import GenerationConfig, AutoConfig
from datasets import Dataset
from utils import batch_iterator

USED_META = ['category', 'template', 'attrs', 'info']
# Preprocessing functions
def preprocess(model, tokenizer, batch, config, device, **kwargs):
    n = len(batch['description'])
    setting = kwargs.pop('model_name', 'produc2query') 
    template = kwargs.pop('template', '{0} ||| {1}') 

    # 1) product2query: fine-tuned product2query template for t5 v1.1
    # 2) t5-cnndm: pre-trained summarization modeles
    # 2) bart-cnndm: pre-trained summarization models like pegasus or bart
    processed_input = tokenizer(
            [template.format(batch['title'][i], batch['description'][i]) for i in range(n)],
            max_length=kwargs.pop('max_src_length', 512),
            truncation=True,
            padding=True,
            return_tensors='pt'
    ).to(device)

    outputs = model.generate(
            **processed_input, 
            generation_config=config
    )
    processed_output = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
    )
    return processed_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--output_jsonl", type=str, default=None)

    parser.add_argument("--model_name", default='t5-base', type=str)
    parser.add_argument("--model_hf_name", default='t5-base', type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_src_length", default=256, type=int)
    parser.add_argument("--max_tgt_length", default=32, type=int)
    parser.add_argument("--num_return_sequences", default=3, type=int)
    parser.add_argument("--template", default=None, type=str)

    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    # load hf 
    if 'product2query' in args.model_name:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_hf_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_hf_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(args.device)
    model = model.eval()

    # load config
    generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name=args.model_hf_name,
            num_beams=args.num_beams, 
            top_k=args.top_k, 
            do_sample=args.do_sample, 
            max_new_tokens=args.max_tgt_length, 
            num_return_sequences=args.num_return_sequences,
    )

    fout = open(args.output_jsonl, 'w')

    # load data
    data = []
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())

            metadata = []
            for k in [m for m in USED_META if m in item.keys()]:
                v = item[k]
                if len(v) != 0:
                    if type(v) == str:
                        continue
                    elif type(v) == list:
                        v = " ".join(v)
                    elif type(v) == dict:
                        v = " ".join(v.values())
                    metadata.append(v)

            data.append({
                    'doc_id': item['doc_id'],
                    'title': item['title'],
                    'description': item['description'], 
                    'metadata': " ".join(metadata)
            })
    dataset = Dataset.from_list(data)
    print(dataset)

    data_iterator = batch_iterator(dataset, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
        batch_ids = batch['doc_id']
        batch_titles = batch['title']
        summarized_texts = preprocess(
                model, tokenizer, batch, 
                generation_config,
                device=args.device,
                max_src_length=args.max_src_length,
                model_name=args.model_name,
                template=args.template
        )

        # enumerate the generated outputs
        for i in range(len(batch_ids)):
            docid = batch_ids[i]
            title = batch_titles[i]
            summarized_text = summarized_texts[i]
            if args.num_return_sequences > 1:
                start = i * args.num_return_sequences
                end = start+args.num_return_sequences
                summarized_text = " ".join(summarized_texts[start: end])

            fout.write(json.dumps({
                "id": str(docid), 
                "contents": title + " " + summarized_text
            })+'\n')

    fout.close()
    print("Done")
