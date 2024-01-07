import os
from tqdm import tqdm
import collections
import argparse
import json
from PIL import Image
from datasets import Dataset
from pyserini.search.lucene import LuceneSearcher

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

def load_images(path):
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            docid = line.strip()
            filename = os.path.join('/home/jhju/datasets/pdsearch/images/', f"{docid}.jpg")
            data[docid] = filename
    print("total available images:", len(data))
    return data

def load_corpus(path='data/corpus.jsonl', append=False, key='title'):
    data = {}
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
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

    print(f"positive qrels {p}")
    print(f"negative qrels {n}")
    return data_pos, data_neg

def complement_negative(pqrels_1, pqrels_0, index_dir, k=300, k1=0.9, b=0.4):
    # load retriever
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1, b)

    # retrieval
    for qid in tqdm(pqrels_1, total=len(pqrels_1)):
        if qid in queries:
            pos_docs = pqrels_1[qid]
            neg_docs = pqrels_0[qid]
            offset = len(pos_docs) - len(neg_docs)

            hits = searcher.search(queries[qid], k=k)
            docids = [hit.docid for hit in hits if hit.docid not in neg_docs]
            boundary = max([i for i, docid in enumerate(docids) if docid in pos_docs] + [0])
            to_add = [docid for docid in docids[boundary:]][:offset]
            pqrels_0[qid] += to_add

    return pqrels_0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=str, default='sample.qrels')
    parser.add_argument("--query", type=str, default='query.tsv')
    parser.add_argument("--index_dir", type=str, default='query.tsv')
    parser.add_argument("--collection", type=str, default='sample.jsonl')
    parser.add_argument("--img_collection", type=str, default='sample.jsonl')
    parser.add_argument("--output", type=str, default='data/trec-pds.train.product2query.jsonl')
    parser.add_argument("--thres", type=int, default=1)
    args = parser.parse_args()

    # load data
    queries = load_query(args.query)
    print('load query: done')
    corpus = load_corpus(args.collection)
    print('load corpus: done')
    images = load_images(args.img_collection)
    print('load images: done')
    pqrels_1, pqrels_0 = load_qrels(args.qrels, args.thres)
    print('load qrels: done') # only used positive ones.

    # BM25 negative sampling to complement
    pqrels_0 = complement_negative(
            pqrels_1, pqrels_0, 
            index_dir=args.index_dir, 
            k=300, k1=0.9, b=0.4
    )

    with open(args.output, 'w') as fout:
        for qid in tqdm(pqrels_1, total=len(pqrels_1)):
            for docid in pqrels_1[qid]:
                try:
                    example = {'query': queries[qid]}
                    example['title'] = corpus[docid]['title']
                    example['image'] = images[docid]
                    example['label'] = 1
                    fout.write(json.dumps(example, ensure_ascii=False)+'\n')
                except:
                    print('missing product or query', docid)

        for qid in tqdm(pqrels_0, total=len(pqrels_0)):
            for docid in pqrels_0[qid]:
                try:
                    example = {'query': queries[qid]}
                    example['title'] = corpus[docid]['title']
                    example['image'] = images[docid]
                    example['label'] = 0
                    fout.write(json.dumps(example, ensure_ascii=False)+'\n')
                except:
                    print('missing product or query', docid)

