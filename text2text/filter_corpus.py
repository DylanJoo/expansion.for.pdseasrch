import os
from tqdm import tqdm
import collections
import argparse
import json

def main(corpus_path, filtered_path, title_only=False):
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

            if title_only:
                contents = title
            else:
                contents = f"{title} {description}"

            if (data_type != 'error'):
                fo.write(json.dumps(
                    {'id': doc_id, 'contents': contents}, ensure_ascii=False
                )+'\n')
            else:
                n_removed +=1

    print(f'Remove {n_removed} documents(products).')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--title_only", action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'corpus.jsonl')
    main(args.input_jsonl, output_path, args.title_only)
