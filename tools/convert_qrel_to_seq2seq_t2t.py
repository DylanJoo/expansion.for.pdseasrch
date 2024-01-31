import os
from tqdm import tqdm
import collections
import argparse
import json

USED_META = ['category', 'template', 'attrs', 'info']
def extract_metadata(item):
    metadata = []
    avail_meta = [m for m in USED_META if m in item.keys()]
    for k in avail_meta:
        v = item[k]
        if len(v) != 0:
            if type(v) == str:
                continue
            elif type(v) == list:
                v = " ".join(v)
            elif type(v) == dict:
                v = " ".join(v.values())
            metadata.append(v)
    return " ".join(metadata)

def load_query(path='data/qid2query.tsv'):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                qid, qtext = line.strip().split('\t')
            except:
                qid = line.strip()
                qtext = ""
                print("Filtered query: {}\t{}".format(qid, qtext))

            qtext_toks = qtext.split()
            if ( len(qtext_toks) == 1) and (qtext_toks[0].startswith("B")):
                print("Filtered query: {}\t{}".format(qid, qtext))
            elif ( len(qtext_toks) == 1) and (qtext_toks[0] == ""):
                print("Filtered query: {}\t{}".format(qid, qtext))
            elif qtext == "":
                print("Filtered query: {}\t{}".format(qid, qtext))
            else:
                data[str(qid.strip())] = qtext
    return data

def load_corpus(path='data/corpus.jsonl', append=False, key='title'):
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            docid = item.pop('doc_id')
            # filter out invalid products
            if (item['title'] + item['description']).strip() != "":
                data[str(docid)] = {
                        'title': item['title'],
                        'description': item['description'], 
                        'metadata': extract_metadata(item)
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
    parser.add_argument("--output", type=str, default='data/trec-pds.train.product2query.jsonl')
    parser.add_argument("--thres", type=int, default=2)
    args = parser.parse_args()

    # load data
    queries = load_query(args.query)
    print('load query: done')
    corpus = load_corpus(args.collection)
    print('load corpus: done')
    pqrels, _ = load_qrels(args.qrels, args.thres)
    print('load qrels: done') # only used positive ones.

    with open(args.output, 'w') as fout:
        for qid in tqdm(pqrels, total=len(pqrels)):
            for docid in pqrels[qid]:
                try:
                    example = {'query': queries[qid]}
                    example['title'] = corpus[docid]['title']
                    example['description'] = corpus[docid]['description']
                    example['metadata'] = corpus[docid]['metadata']
                    fout.write(json.dumps(example, ensure_ascii=False)+'\n')
                except:
                    print('missing query or product', docid)

