import re
from tqdm import tqdm
import json
import collections
import warnings

import torch
import numpy as np
import collections

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

def batch_transform_token_ids(tokenizer, batch_token_ids, return_attention_mask=False):
    tokens_to_add = []
    string_to_return = []
    
    # token_ids --> a token list and a sring
    for token_ids in batch_token_ids:
        decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_string = tokenizer.convert_tokens_to_string(decoded_tokens)
        tokens_to_add += decoded_tokens
        string_to_return.append(decoded_string)
        
    # strings --> offset_mapping
    tokenized = tokenizer(
            string_to_return,
            add_special_tokens=False, return_offsets_mapping=True,
            padding=True, return_tensors='pt'
    )       
    
    ## [IMPORTANT] this is very ad-hoc way to deal with destriptive process of encoding
    if tokenized.input_ids.size(1) != batch_token_ids.size(1):
        tokenizer.add_tokens(list(set(tokens_to_add)))
        tokenized = tokenizer(
                string_to_return,
                add_special_tokens=False, return_offsets_mapping=True,
                padding=True, return_tensors='pt'
        )       
        mapping_to_return = tokenized.offset_mapping
    else:
        mapping_to_return = tokenized.offset_mapping
        
    # check length
    if return_attention_mask:
        # mask = tokenized.attention_mask # derived from output token ids
        mask = (batch_token_ids != tokenizer.pad_token_id).long()
        return string_to_return, mapping_to_return, mask
    else:
        return string_to_return, mapping_to_return, None
        
def batch_map_word_values(logits, batch_token_ids, strings, offset_mapping, is_pooled=False):

    if is_pooled:
        logits = logits.unsqueeze(1).repeat((1, batch_token_ids.size(-1), 1))
    weights = logits.gather(2, batch_token_ids.unsqueeze(2)).squeeze(2).cpu().numpy()

    to_return = []
    for i, (offset, weight) in enumerate(zip(offset_mapping, weights)):
        word_id_to_weights = collections.defaultdict(float)
        words = strings[i]

        start, end = 0, 0
        for j, (start_, end_) in enumerate(offset):
            # retrieve the maximum between tokens
            if end_ != 0:
                w = weight[j]
                if start_ == end:
                    prev = word_id_to_weights[words[start:end]]
                    del word_id_to_weights[words[start:end]]
                    start, end = start, end_
                    word_id_to_weights[words[start: end]] = prev # max(prev, w)

                else: # separated
                    start, end = start_, end_
                    word_id_to_weights[words[start: end]] = w

        word_id_to_weights.pop('[SEP]', None) # remove spec token
        word_id_to_weights.pop('[CLS]', None) # remove spec token
        word_id_to_weights.pop('[PAD]', None) # remove spec token
        to_return.append(word_id_to_weights)
    return to_return

