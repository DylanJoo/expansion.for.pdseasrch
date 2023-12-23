import os
import re
import random
from tqdm import tqdm
import json
import collections
import warnings
from PIL import Image
import requests

def load_query(path='/tmp2/trec/pds/data/query/qid2query.tsv'):
    data = collections.defaultdict(str)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            qid, qtext = line.split('\t')
            data[str(qid.strip())] = qtext.strip()
    return data

def load_images(path):
    data = collections.defaultdict(str)
    # data_dir = os.path.join(path.rsplit('/', 1)[0], 'collection')
    with open(path, 'r') as f:
        for line in tqdm(f):
            docid = line.strip()
            filename = os.path.join('/home/jhju/datasets/pdsearch/images/', f"{docid}.jpg")
            data[docid] = filename
    print("total available images:", len(data))
    return data

def load_title(path='/tmp2/trec/pds/data/collection/collection_sim_title.jsonl'):
    data = dict()
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        # [bug] valule `docid` is inccoret
        doc_id = item.pop('id')
        contents = item.pop('contents')
        data[str(doc_id)] = contents
    return data

def load_collection(path, append=False):
    data = {}
    # data = collections.defaultdict(lambda: 'NA')
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        # [bug] valule `docid` is inccoret
        doc_id = item.pop('doc_id')
        if append:
            title = item['title']
            asin = item.get('asin', "")
            title = f"{title} {asin}".strip()
            description = item['description']
            data[str(doc_id)] = f"{title}{append}{description}"
        else:
            data[str(doc_id)] = {
                    'title': item['title'], 
                    'description': item['description']
            }
    return data

def load_qrel(path='/tmp2/trec/pds/data/qrels/product-search-train.qrels', thres=1):
    positives = collections.defaultdict(list)
    negatives = collections.defaultdict(list)
    fi = open(path, 'r')
    for line in tqdm(fi):
        qid, _, docid, relevance = line.strip().split('\t')
        if int(relevance) >= thres:
            positives[qid] += [docid] # greater the better
        else:
            negatives[qid] += [docid] # greater the better
    return positives, negatives

def load_run(path='/tmp2/trec/pds/data/qrels/pyserini-full-train-2023.run', top_k=10000):
    data = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= top_k:
                data[qid] += [(docid, float(rank))]

    sorted_data = collections.OrderedDict()
    for (qid, docid_list) in tqdm(data.items()):
        sorted_docid_list = sorted(docid_list, key=lambda x: x[1]) 
        sorted_data[qid] = [docid for docid, _ in sorted_docid_list]
    return sorted_data

def load_qp_pair(path='/tmp2/trec/pds/data/qrels/pyserini-full-train-2023.run', top_k=10000):
    data = {'qid': [], 'docid': []}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= top_k:
                data['qid'].append(qid)
                data['docid'].append(docid)
    return data

def expand_collection(path_base, path_expand, output_jsonl='expanded.jsonl'):
    original = load_collection(path=path_base, append=False)
    expansion = load_collection(path=path_expand, append=False)
    with open(output_jsonl, 'w') as f:
        for docid in tqdm(original):
            expanded_contents = expansion.get(docid, "")
            if isinstance(expanded_contents, list):
                expanded_contents = " ".join(expanded_contents)
            contents = original[docid] + " " + expanded_contents
            example = {"id": docid, "contents": contents}
            f.write(json.dumps(example)+'\n')

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def random_mask(x, drop_p=0.5):
    spec = "[MASK]"
    w = x.split()
    drop = random.choices([0,1], k=len(w), weights=[drop_p, 1-drop_p])
    w = [ww if l==1 else spec for ww, l in zip(w, drop)]
    return " ".join(w)
