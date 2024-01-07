import os
import torch
from tqdm import tqdm 
import argparse
from transformers import GenerationConfig, AutoConfig
from tools import batch_iterator

def inference(model, processor, queries, config, device, **kwargs):
    texts = [f"Query: {q}" for q in queries]

    processed_input = processor(
        text=texts,
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
    parser.add_argument("--query", type=str, default="data/qid2query-dev-filtered.tsv")
    parser.add_argument("--output_tsv", type=str, default=None)
    parser.add_argument("--model_name", default='t5-base', type=str)
    parser.add_argument("--model_hf_name", default='t5-base', type=str)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_src_length", default=256, type=int)
    parser.add_argument("--max_tgt_length", default=32, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    # load hf 
    from models import BlipForQuestionAnswering
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
    )

    fout = open(args.output_tsv, 'w')

    data_list = []
    with open(args.query, 'r') as f:
        for line in tqdm(f):
            qid, qtext = line.strip().split('\t')
            data = {'id': qid, 'text': qtext}
            data_list.append(data)

    # inference:  
    data_iterator = batch_iterator(data_list, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(data_list)//args.batch_size+1):
        batch_ids = [b['id'] for b in batch]
        batch_queries = [b['text'] for b in batch]
        batch_queries_rewritten = inference(
                model=model, 
                processor=processor,
                queries=batch_queries,
                config=generation_config,
                device=args.device,
                max_src_length=args.max_src_length
        )

        for qid, qtext, qtext_orig in zip(
                batch_ids, batch_queries_rewritten, batch_queries
        ):
            fout.write(f"{qid}\t{qtext_orig} {qtext}\n")

    print("Done")
