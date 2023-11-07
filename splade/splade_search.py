"""
[TODO] Put this into sparse search
"""
import os
from tqdm import tqdm 
import json
import argparse
from pyserini.search.lucene import LuceneImpactSearcher
from utils import load_collection, load_query

def search(args):
    searcher = LuceneImpactSearcher(args.index, args.encoder, args.min_idf)

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
            hits = searcher.batch_search(
                    queries=qtexts_batch, 
                    q_ids=qids_batch, 
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
    parser.add_argument("--output", default='../runs/run.sample.txt', type=str)
    # special args
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    search(args)
    print("Done")
