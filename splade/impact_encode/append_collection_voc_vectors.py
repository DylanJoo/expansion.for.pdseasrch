import json
import os
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from naver_splade.splade.models.transformer_rep import Splade
from collections import defaultdict
import sys
from datasets import load_dataset

sys.path.insert(0, "/tmp2/jhju/codes.trec.productsearch2023/codes/")
from utils import batch_iterator

def generate_vocab_vector(docs, model, minimum=0, device='cpu', max_length=256, quantization_factor=1000):
    """
    params: docs: List[str]
    returns: vectors: List[Dict]
    """
    # now compute the document representation
    inputs = tokenizer(docs, return_tensors="pt", 
            padding='max_length', truncation=True, max_length=256
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        doc_reps = model(d_kwargs=inputs)["d_rep"]  
        # (sparse) doc rep in voc space, shape (30522,)

    # get the number of non-zero dimensions in the rep:
    cols = torch.nonzero(doc_reps)

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    # sort them 
    def sort_dict(dictionary, quantization_factor):
        d = {k: v*quantization_factor for (k, v) in dictionary if v >= minimum}
        sorted_d = {reverse_voc[k]: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        return sorted_d

    return [sort_dict(weight, quantization_factor) for i, weight in weights.items()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default='/tmp2/trec/pds/data/collection/collection_sim.jsonl')
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--model_name_or_dir", type=str, default="/tmp2/trec/pds/models/distilsplade_max")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--minimum", type=float, default=0)
    args = parser.parse_args()

    # load models
    model = Splade(args.model_name_or_dir, agg="max")
    model.transformer_rep.transformer.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir or args.tokenizer_name)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    # load collection
    collection = load_dataset('json', data_files=args.collection)['train']
    n_collection = len(collection)
    print(collection)

    # outout writer
    os.makedirs(args.collection_output.rsplit('/', 1)[0], exist_ok=True)
    fout = open(args.collection_output, 'w')

    # preparing batch 
    vectors = []
    for (start, end) in tqdm(
            batch_iterator(collection, args.batch_size, return_index=True),
            total=(n_collection//args.batch_size)+1
    ):
        batch_collection = collection[start: end]
        batch_vectors = generate_vocab_vector(
                docs=batch_collection['contents'], 
                model=model,
                minimum=args.minimum,
                device=args.device,
                max_length=args.max_length,
                quantization_factor=args.quantization_factor
        )
        assert len(batch_collection['contents']) == len(batch_vectors), \
                'Mismatched amount of examples'

        vectors += batch_vectors

    # collection and re-dump the collections
    for i, example in enumerate(collection):
        example.update({"vector": vectors[i]})

        fout.write(json.dumps(example, ensure_ascii=False)+'\n')

    fout.close()

