import requests
from tqdm import tqdm
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

IMGLIST='/home/jhju/datasets/pdsearch/corpus-imgs.jsonl'
GALLERY='images/'

def download(data):
    doc_id = data['id']
    filename = os.path.join('images', data['contents'])
    img_url = data.pop('url', None)

    # img=Image.open(filename)
    # img.verify()
    if (os.path.exists(filename) is False) and (img_url is not None):
        try:
            img_content = requests.get(img_url).content
        except:
            # print("download failed:", doc_id, img_url)
            img_content = None

        if img_content is not None:
            fo = open(filename, 'wb')
            fo.write(img_content)
            fo.close()

data_list = []
with open(IMGLIST, 'r') as fi:
    for line in tqdm(fi):
        data = json.loads(line.strip())
        data_list.append(data)

with ThreadPoolExecutor(max_workers=40) as executor:
    executor.map(download, data_list) 
