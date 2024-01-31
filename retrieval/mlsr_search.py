"""
[TODO] Put this into sparse search
"""
import os
import collections
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneImpactSearcher
from encode import SpladeQueryEncoder, SpladeQueryLexicalEncoder

def load_query(path):
    data = collections.defaultdict(str)
    with open(path, 'r') as f:
        for line in f:
            if path.endswith('tsv'):
                qid, qtext = line.split('\t')
                data[str(qid.strip())] = qtext.strip()
            else:
                raise ValueError("Invalid data extension.")
    return data

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def search(args):
    if args.use_lexical:
        query_encoder = SpladeQueryLexicalEncoder(
                args.encoder, 
                device=args.device, 
                mask_appeared_tokens=False if args.include_both else True,
                gamma_token=1,
                gamma_word=1.5
        )
    else:
        query_encoder = SpladeQueryEncoder(
                args.encoder, device=args.device
        )
        # query_encoder = BlipForQueryEncoder.from_pretrained(
        #         args.encoder, 
        #         processor_name=args.processor, 
        #         pooling="max"
        # )
    query_encoder.model.eval()
    searcher = LuceneImpactSearcher(args.index, query_encoder, args.min_idf)

    # for example
    query = load_query(args.query)
    qids = list(query.keys())
    qtexts = list(query.values())

    # prepare the output file
    output = open(args.output, 'w')

    # search for each q
    if args.batch_size == 1:
        for qid, qtext in tqdm(query.items()):
            hits = searcher.search(qtext, k=args.k)
            for i in range(len(hits)):
                output.write(f'{qid} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} SPLADE\n')
    else:
        for (start, end) in tqdm(
                batch_iterator(range(0, len(qids)), args.batch_size, return_index=True),
                total=(len(qids)//args.batch_size)+1
            ):
            qids_batch = qids[start: end]
            qtexts_batch = qtexts[start: end]
            # for qtexts in qtexts_batch:
            #     print(query_encoder.encode(qtexts))
            hits = searcher.batch_search(
                    queries=qtexts_batch, 
                    qids=qids_batch, 
                    k=args.k
            )
            for key, value in hits.items():
                for i in range(len(hits[key])):
                    output.write(f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} SPLADE\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--min_idf",type=float, default=0)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--processor", type=str)
    parser.add_argument("--output", default='../runs/run.sample.txt', type=str)
    # special args
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--use_lexical", action='store_true', default=False)
    parser.add_argument("--include_both", action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    search(args)
    print("Done")
