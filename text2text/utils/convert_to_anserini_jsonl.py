import os
from tqdm import tqdm
import json
import argparse
from tools import load_collection

USED_TEXT = ['contents', 'title', 'description', 'asin']
USED_META = ['category', 'template', 'attrs', 'info']

def load_collection(path):
    data = dict()
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        # [bug] valule `docid` is inccoret
        doc_id = item.pop('id')
        data[str(doc_id)] = item['contents'].strip()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=str, required=True)
    parser.add_argument('--fields', nargs='+', default=['title'], required=False)
    parser.add_argument('--full', action='store_true', default=False)
    parser.add_argument('--output_jsonl', type=str, required=True, default='doc00.jsonl')
    args = parser.parse_args()

    os.makedirs(args.output_jsonl.rsplit('/', 1)[0], exist_ok=True)
    fo = open(args.output_jsonl, 'w')

    for field in args.fields:
        assert field in USED_TEXT, f'Invalid field {field}'

    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            try:
                docid = data.pop('doc_id')
            except:
                docid = data.pop('id')

            # the simplifed version
            contents = ""
            keys = [t for t in args.fields if t in data.keys()]
            for k in keys:
                v = data[k]
                contents += f" {v}"

            # the full version
            # add meta into contents
            if args.full:
                contents += " ||| "
                keys = [m for m in USED_META if m in data.keys()]
                for k in keys:
                    v = data[k]
                    if len(v) != 0:
                        if type(v) == str:
                            contents += f" {v}"
                        elif type(v) == list:
                            v = " ".join(v)
                            contents += f" {v}"
                        elif type(v) == dict:
                            v = " ".join(v.values())
                            contents += f" {v}"

            contents = contents.replace('\\', '')
            fo.write(json.dumps({"id": docid, "contents": contents}, ensure_ascii=False)+'\n')

