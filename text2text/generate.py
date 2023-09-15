import os
import torch
from tqdm import tqdm 
import json
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GenerationConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

# t5 product to query
def summarize_p2q(model, tokenizer, batch, config, device, **kwargs):
    n = len(batch['description'])
    processed_input = tokenizer(
            [f"summarize: title: {batch['title'][i]} description: {batch['description'][i]}" for i in range(n)],
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
    parser.add_argument("--tokenizer_name", default='t5-base', type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_src_length", default=256, type=int)
    parser.add_argument("--max_tgt_length", default=32, type=int)
    parser.add_argument("--num_return_sequences", default=3, type=int)

    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    assert 't5' in args.model_name, "t5 only, so far"
    assert 't5' in args.tokenizer_name, "t5 only, so far"

    # load hf 
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(args.device)
    model = model.eval()

    # load config
    generation_config = GenerationConfig.from_pretrained(
            "t5-base", 
            num_beams=args.num_beams, 
            top_k=args.top_k, 
            do_sample=args.do_sample, 
            max_new_tokensargs.max_tgt_length, 
            num_return_sequences=args.num_return_sequences 
    )
    output_jsonl = args.collection.replace('.jsonl', '.predicted.jsonl')
    output_jsonl = (args.output_jsonl or output_jsonl)

    fout = open(output_jsonl, 'w')

    # load data
    dataset = load_dataset('json', data_files=args.collection)['train']
    dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            pin_memory=True, 
            drop_last=False, 
    )
    for batch in tqdm(dataloader):
        if 'product2query' in args.model_name:
            summarized_texts = summarize_p2q(
                    model, tokenizer, batch, 
                    generation_config,
                    device=args.device,
                    max_src_length=args.max_src_length
            )
        batch_ids = batch['id'].detach().numpy()

        # if 'desc2title' in args.model_name:
        #     summarized_texts = summarize_d2t(
        #             model, tokenizer, batch, 
        #             generation_config,
        #             device=args.device,
        #             max_src_length=args.max_src_length
        #     )

        # enumerate the generated outputs
        for i in range(len(batch_ids)):
            docid = batch_ids[i]
            summarized_text = summarized_texts[i]
            if args.num_return_sequences > 1:
                start = i * args.num_return_sequences
                end = start+args.num_return_sequences
                summarized_text = summarized_texts[start: end]

            fout.write(json.dumps({
                "id": str(docid), "contents": summarized_text
            })+'\n')

    fout.close()
    print("Done")

# # t5 desc to title
# def summarize_d2t(model, tokenizer, batch, config, device, **kwargs):
#     n = len(batch['description'])
#     processed_input = tokenizer(
#             [f"summarize: {batch['description'][i]}" for i in range(n)],
#             max_length=kwargs.pop('max_src_length', 512),
#             truncation=True,
#             padding=True,
#             return_tensors='pt'
#     ).to(device)
#
#     outputs = model.generate(
#             **processed_input, 
#             generation_config=config
#     )
#     processed_output = tokenizer.batch_decode(
#             outputs, skip_special_tokens=True
#     )
#     # remove the texts without desc
#     for j in [i for i in range(n) if batch['description'][i] == ""]:
#         processed_output[j] = ""
#     return processed_output
