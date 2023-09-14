import os
from tqdm import tqdm
import collections
import argparse
import json
from tools import load_title, load_run, load_query

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--truncate", type=int, default=1e10)
    args = parser.parse_args()

    titles = load_title()
    runs = load_run(args.run)
    queries = load_query()

    print(args.run)
    filename = args.run.replace('run', 'log')
    fout = open(filename, 'w')

    for i, (qid, doc_list) in enumerate(runs.items()):
        if i > args.truncate:
            break
        query = queries[qid]
        fout.write(f"* {query}\n")
        for docid in doc_list[:args.topk]:
            title = titles.get(docid, f"NA:{docid}")
            fout.write(f"# {title}\n")
        fout.write("\n")


