import json
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from naver_splade.splade.models.transformer_rep import Splade
from collections import defaultdict
from datasets import Dataset
from utils import batch_iterator

def generate_vocab_vector(
    docs, 
    model, 
    minimum=0, 
    device='cpu', 
    max_length=256, 
    quantization_factor=1000,
):
    """
    params: docs: List[str]
    returns: vectors: List[Dict]
    """
    # now compute the document representation
    inputs = tokenizer(docs, 
                       return_tensors="pt", 
                       padding='max_length', 
                       truncation=True, 
                       max_length=256,
                       return_offsets_mapping=False)
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
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--model_name_or_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # load models and tokenizer
    model = Splade(args.model_name_or_dir, agg="max")
    model.transformer_rep.transformer.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir or args.tokenizer_name)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    # load data
    data_list = []
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data_list.append(item)

            if len(data_list) >= 100 and args.debug:
                break

    dataset = Dataset.from_list(data_list)
    print(dataset)

    # preparing batch 
    vectors = []
    data_iterator = batch_iterator(dataset, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
        batch_vectors = generate_vocab_vector(
                docs=batch['contents'], 
                model=model,
                minimum=args.minimum,
                device=args.device,
                max_length=args.max_length,
                quantization_factor=args.quantization_factor,
        )
        assert len(batch['contents']) == len(batch_vectors), \
                'Mismatched amount of examples'

        vectors += batch_vectors

    # outout writer
    fout = open(args.collection_output, 'w')

    # collection and re-dump the collections
    for i, example in enumerate(data_list):
        example.update({"vector": vectors[i]})
        fout.write(json.dumps(example, ensure_ascii=False)+'\n')

    fout.close()

