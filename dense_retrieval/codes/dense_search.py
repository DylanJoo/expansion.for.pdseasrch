import os
import torch
from tqdm import tqdm
import argparse
from pyserini.search import FaissSearcher
from utils import load_query, batch_iterator

def search(args):
        
    searcher = FaissSearcher(args.index, args.encoder_name)

    if torch.cuda.is_available():
        searcher.query_encoder.model.to(args.device)
        searcher.query_encoder.device = args.device

    query = load_query(args.query)
    qids = list(query.keys())
    qtexts = list(query.values())

    # prepare the output file
    output = open(args.output, 'w')

    # search for each q
    if args.batch_size == 1:
        for qid, qtext in tqdm(query.items()):
            hits = searcher.search(qtext, k=args.k, threads=10)
            for i in range(len(hits)):
                output.write(f'{qid} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} FAISS\n')
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
            for key, _ in hits.items():
                for i in range(len(hits[key])):
                    output.write(f'{key} Q0 {hits[key][i].docid:4} {i+1} {hits[key][i].score:.5f} FAISS\n')
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--index", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--encoder_name", type=str)
    # special args
    parser.add_argument("--query", default=None, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    search(args)
    print("Done")
