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

def load_query_with_llm(path='/tmp2/trec/pds/data/query/pds2023-dev-llm4qe.jsonl', topk=1, keep_orig=True):
    data = dict()
    with open(path, 'r') as f:
        for line in f:
            if "llm4qe" in path.lower():
                item = json.loads(line.strip())
                qtext = item['query']
                # postprocess the semicolon-separated wordlist
                qtext_qe = item['llm_generated_texts'].split(";")[:topk]
                if isinstance(qtext_qe, list):
                    qtext_qe = " ".join(qtext_qe)
                qtext_qe = re.sub("\s+", " ", qtext_qe)
                if keep_orig:
                    data[str(item['id'])] = f"{qtext} {qtext_qe}"
                else:
                    data[str(item['id'])] = qtext_qe

            elif "llm4qr" in path.lower():
                item = json.loads(line.strip())
                qtext = item['query']
                # postprocess the semicolon-separated wordlist
                qtext_qr = item['llm_generated_texts']['rewritten']
                qtext_others = item['llm_generated_texts']['others']
                if keep_orig:
                    data[str(item['id'])] = f"{qtext} {qtext_qr}"
                else:
                    data[str(item['id'])] = qtext_qr
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

def monot5_preprocess(example, query, documents):
    q = query[example['qid']]
    d = documents[example['docid']]
    example['source'] = f"Query: {q} Document: {d} Relevant:"
    return example

def minilm_preprocess(example, query, documents):
    q = query[example['qid']]
    d = documents[example['docid']]
    example['query'] = q
    example['document'] = d
    return example

def query_rewrite(x):
    return f'Generate a Dictionary with requested keys "brand", "category", "feature", and "type", and "human-readable query" for the query: "{x}"'

def document_extraction(x):
    return f'Generate a List of 10 relevant keywords for the given product description: "{x}"'
