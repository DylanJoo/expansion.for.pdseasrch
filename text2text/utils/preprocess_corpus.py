import os
from tqdm import tqdm
import collections
import argparse
import requests
import json

# Collection
CORPUS='/tmp2/trec/pds/data/collection/collection_full.jsonl'
def get_simple_collection():
    CORPUS_SIM='/tmp2/trec/pds/data/collection/collection_sim.jsonl'
    fo = open(CORPUS_SIM, 'w')
    with open(CORPUS, 'r') as fi:
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data.pop('doc_id')
            title = data.pop('title', '')
            description = data.pop('description', '')
            asin = data.pop('asin', '')

            if data['type'] != 'error':
                fo.write(json.dumps({
                    'id': doc_id, 
                    'title': title, 
                    'description': description, 
                    'asin': asin
                }, ensure_ascii=False)+'\n')
            elif asin != '':
                fo.write(json.dumps({
                    'id': doc_id, 
                    'title': "", 
                    'description': "", 
                    'asin': asin
                }, ensure_ascii=False)+'\n')
            else:
                continue

# Collection [TODO]
CORPUS='/tmp2/trec/pds/data/collection/collection_full.jsonl'
def get_full_collection():
    CORPUS_SIM='/tmp2/trec/pds/data/collection/collection_sim.jsonl'
    fo = open(CORPUS_SIM, 'w')
    with open(CORPUS, 'r') as fi:
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data.pop('doc_id')
            title = data.pop('title', '')
            description = data.pop('description', '')
            asin = data.pop('asin', '')

            if data['type'] != 'error':
                fo.write(json.dumps({
                    'id': doc_id, 
                    'title': title, 
                    'description': description, 
                    'asin': asin
                }, ensure_ascii=False)+'\n')

# IMGLIST='/tmp2/trec/pds/data/collection/collection_imgs.json'
IMGLIST='/tmp2/trec/pds/data/collection/collection_images.jsonl'
GALLERY='/tmp2/trec/pds/data/images/'
def download_image_collection():
    with open(IMGLIST, 'r') as fi:
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data['id']
            img_url = data.pop('url', None)
            filename = os.path.join(GALLERY, f"{doc_id}.jpg")
            if (img_url is not None) and (os.path.exists(filename) is False):
                try:
                    img_content = requests.get(img_url).content
                except:
                    img_content = None
                    print(f"doc image: {doc_id} not found. The url is {doc_id}")

                if img_content is not None:
                    fo = open(filename, 'wb')
                    fo.write(img_content)
                    fo.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False)
    parser.add_argument("--get_simple_collection", action='store_true', default=False)
    parser.add_argument("--download_image_collection", action='store_true', default=False)
    args = parser.parse_args()

    if args.get_simple_collection:
        get_simple_collection()

    if args.download_image_collection:
        os.makedirs(GALLERY, exist_ok=True)
        download_image_collection()
