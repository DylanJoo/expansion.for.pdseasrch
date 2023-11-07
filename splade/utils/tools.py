import re
from tqdm import tqdm
import json
import collections
import warnings

def load_query(path='/tmp2/trec/pds/data/query/qid2query.tsv'):
    data = collections.defaultdict(str)
    with open(path, 'r') as f:
        for line in f:
            if path.endswith('tsv'):
                qid, qtext = line.split('\t')
                data[str(qid.strip())] = qtext.strip()
            else:
                raise ValueError("Invalid data extension.")
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

def load_collection(path='/tmp2/trec/pds/data/collection/collection_sim.jsonl', append=False, key='title'):
# def load_collection(path='/tmp2/trec/pds/data/collection/collection_full.jsonl', append=False):
    data = collections.defaultdict(str)
    # data = collections.defaultdict(lambda: 'NA')
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        # [bug] valule `docid` is inccoret
        doc_id = item.pop('id')
        if append:
            title = item['title']
            asin = item.get('asin', "")
            title = f"{title} {asin}".strip()
            description = item['description']
            data[str(doc_id)] = f"{title}{append}{description}"
        else:
            if 'contents' in item:
                key = 'contents'
            data[str(doc_id)] = item[key]
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

def load_run(path='/tmp2/trec/pds/data/qrels/pyserini-full-train-2023.run', topk=10000):
    data = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                data[qid] += [(docid, float(rank))]

    sorted_data = collections.OrderedDict()
    for (qid, docid_list) in tqdm(data.items()):
        sorted_docid_list = sorted(docid_list, key=lambda x: x[1]) 
        sorted_data[qid] = [docid for docid, _ in sorted_docid_list]
    return sorted_data

def load_qp_pair(path='/tmp2/trec/pds/data/qrels/pyserini-full-train-2023.run', topk=10000):
    data = {'qid': [], 'docid': []}
    with open(path, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                data['qid'].append(qid)
                data['docid'].append(docid)
    return data

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

