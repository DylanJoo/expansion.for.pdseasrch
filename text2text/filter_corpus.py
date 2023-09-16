import os
from tqdm import tqdm
import collections
import argparse
import json

def main(corpus_path, filtered_path):
    fo = open(filtered_path, 'w')
    with open(corpus_path, 'r') as fi:
        n_removed = 0
        for line in tqdm(fi):
            data = json.loads(line.strip())
            doc_id = data.pop('doc_id')
            title = data.pop('title', '')
            description = data.pop('description', '')
            data_type = data.pop('type', '')
            asin = data.pop('asin', '')

            if (data_type != 'error'):
                fo.write(json.dumps({
                    'id': doc_id, 
                    'contents': f"{title} {description}",
                }, ensure_ascii=False)+'\n')
            else:
                n_removed +=1

    print(f'Remove {n_removed} documents(products).')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_jsonl", type=str)
    args = parser.parse_args()

    main(args.input_jsonl, args.output_jsonl)
