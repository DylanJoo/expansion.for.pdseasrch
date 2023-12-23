import os
from tqdm import tqdm
import collections
import argparse
import json
from PIL import Image
from datasets import Dataset

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
    invalid_counts = 0
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
            else:
                invalid_counts += 1
    print("Total available texts:", len(data), "Filtered:", invalid_counts) 
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default='query.tsv')
    parser.add_argument("--collection", type=str, default='sample.jsonl')
    parser.add_argument("--img_collection", type=str, default='sample.jsonl')
    parser.add_argument("--output", type=str, default='data/trec-pds.train.product2query.jsonl')
    args = parser.parse_args()

    # load data
    corpus = load_corpus(args.collection)
    print('load corpus: done')
    images = load_images(args.img_collection)
    print('load images: done')

    with open(args.output, 'w') as fout:
        for docid in tqdm(images, total=len(images)):
            try:
                example = {}
                example['title'] = corpus[docid]['title']
                example['description'] = corpus[docid]['description']
                example['image'] = images[docid]
                fout.write(json.dumps(example, ensure_ascii=False)+'\n')
            except:
                print('missing product', docid)

