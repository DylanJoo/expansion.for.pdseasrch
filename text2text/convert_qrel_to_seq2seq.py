import os
from tqdm import tqdm
import collections
import argparse
import requests
import json
from utils import load_run

def load_query(path='data/qid2query.tsv'):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.split('\t')
            data[str(qid.strip())] = qtext.strip()
    return data

def load_collection(path='data/collection_sim.jsonl', append=False, key='title'):
    data = collections.defaultdict(str)
    # data = collections.defaultdict(lambda: 'NA')
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            # [bug] valule `docid` is inccoret
            doc_id = item.pop('id')
            title = item['title']
            description = item['description']
            data[str(doc_id)] = f"{title}{append}{description}"
            else:
                if 'contents' in item:
                    key = 'contents'
                data[str(doc_id)] = item[key]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False)
    parser.add_argument("--get_simple_collection", action='store_true', default=False)
    args = parser.parse_args()

