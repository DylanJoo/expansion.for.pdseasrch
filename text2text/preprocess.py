import os
from tqdm import tqdm
import collections
import argparse
import requests
import json

def expand_collection_p2q():
    from utils import expand_collection
    ROOT='/tmp2/trec/pds/data/'
    expand_collection(
            path_base=f'{ROOT}/collection/collection_sim_title.jsonl',
            path_expand=f'{ROOT}/predictions/product2query.predicted.jsonl',
            output_jsonl=f'{ROOT}/collection/lucene/collection_sim_title_p2q.jsonl'
    )

# Collection
CORPUS='/tmp2/trec/pds/data/collection/collection_full.jsonl'
def get_simple_collection():
    CORPUS_SIM='/tmp2/trec/pds/data/collection/collection_sim.jsonl'
    fo = open(CORPUS_SIM, 'w')
    with open(CORPUS, 'r') as fi:
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data['doc_id']
            title = data.pop('title', '')
            description = data.pop('description', '')

            if data['type'] != 'error':
                fo.write(json.dumps({'id': doc_id, 'title': title, 'description': description}, 
                    ensure_ascii=False)+'\n')

# IMGLIST='/tmp2/trec/pds/data/collection/collection-imgs.json'
# GALLERY='/tmp2/trec/pds/data/images/'
# def download_image_collection():
#     with open(IMGLIST, 'r') as fi:
#         for line in tqdm(fi):
#             data = json.loads(line.strip())
#             doc_id = data['doc_id']
#             img_url = data.pop('image_url', None)
#             filename = os.path.join(GALLERY, f"{doc_id}.jpg")
#             if (img_url is not None) and (os.path.exists(filename) is False):
#                 try:
#                     img_content = requests.get(img_url).content
#                 except:
#                     img_content = None
#
#                 if img_content is not None:
#                     fo = open(filename, 'wb')
#                     fo.write(img_content)
#                     fo.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=False)
    parser.add_argument("--get_simple_collection", action='store_true', default=False)
    args = parser.parse_args()

