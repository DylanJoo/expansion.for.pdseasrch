import json
import argparse
from tqdm import tqdm
import numpy as np
from pyserini.search.lucene import LuceneSearcher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--output", default='sample.jsonl', type=str)
    parser.add_argument("--index_dir", type=str, default=None)
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--k1",default=0.9, type=float)
    parser.add_argument("--b", default=0.4, type=float)
    args = parser.parse_args()


    # pyserini
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(k1=args.k1, b=args.b)

    # batch search
    n = len(open(args.query, 'r').readlines())
    with open(args.query, 'r') as fin, open(args.output, 'w') as fout:
        for line in tqdm(fin, total=n):
            qid, qtext = line.strip().split('\t')
            hits = searcher.search(qtext, k=args.k)
            for i, hit in enumerate(hits):
                fout.write(f"{qid} Q0 {hit.docid} {i+1} {hit.score} pyserini\n")
    print('done')
