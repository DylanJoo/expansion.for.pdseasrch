import os
from tqdm import tqdm
import collections
import argparse
import json

def load_query(path='data/qid2query.tsv'):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.split('\t')
            data[str(qid.strip())] = qtext.strip()
    return data

def load_corpus(path='data/corpus.jsonl', append=False, key='title'):
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            docid = item.pop('doc_id')
            data[docid] = {
                    'title': item['title'],
                    'description': item['description']
            }
    return data

def load_qrels(path='data/product-search-train.qrels', thres=1):
    data_pos = collections.defaultdict(list)
    data_neg = collections.defaultdict(list)
    p, n = 0, 0
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, relevance = line.strip().split('\t')
            if int(relevance) >= thres:
                data_pos[qid] += [docid] 
                p += 1
            else:
                data_neg[qid] += [docid]
                n += 1
    print(f"Total queries {len(data_pos)}")
    print(f"positive qrels {p}")
    print(f"negative qrels {n}")
    return data_pos, data_neg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=str, default='sample.qrels')
    parser.add_argument("--query", type=str, default='query.tsv')
    parser.add_argument("--collection", type=str, default='sample.jsonl')
    parser.add_argument("--output", type=str, default='trec-pds.train.product2query.jsonl')
    args = parser.parse_args()

    # load data
    queries = load_query(args.query)
    print('load query: done')
    corpus = load_corpus(args.collection)
    print('load corpus: done')
    pqrels, _ = load_qrels(args.qrels, 2)
    print('load qrels: done')

    with open(args.output, 'w') as fout:
        for qid in tqdm(pqrels, total=len(pqrels)):
            for docid in pqrels[qid]:
                fout.write(json.dumps({
                    "query": queries[qid],
                    "title": corpus[docid]['title'],
                    "description": corpus[docid]['description'],
                }, ensure_ascii=False)+'\n')

