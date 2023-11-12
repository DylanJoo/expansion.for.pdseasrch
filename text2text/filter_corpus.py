import os
from tqdm import tqdm
import collections
import argparse
import json

USED_META = ['category', 'template', 'attrs', 'info']
def main(corpus_path, filtered_path, setting='title'):
    fo = open(filtered_path, 'w')
    with open(corpus_path, 'r') as fi:
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data.pop('doc_id')
            title = data.pop('title', '')
            description = data.pop('description', '')
            # data_type = data.pop('type', '')

            metadata = []
            for k in [m for m in USED_META if m in data.keys()]:
                v = data[k]
                if len(v) != 0:
                    if type(v) == str:
                        continue
                    elif type(v) == list:
                        v = " ".join(v)
                    elif type(v) == dict:
                        v = " ".join(v.values())
                    metadata.append(v)
            metadata = " ".join(metadata)

            # title
            # simplified (title+desc.)
            # full (title+desc.+metadata)
            if setting == 'title':
                contents = title
            elif setting == 'simplified':
                contents = f"{title} {description}"
            else:
                contents = f"{title} {description} {metadata}"

            fo.write(json.dumps(
                {'id': doc_id, 'contents': contents}, ensure_ascii=False
            )+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--setting", type=str, default='title')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'corpus.jsonl')
    main(args.input_jsonl, output_path, args.setting)
