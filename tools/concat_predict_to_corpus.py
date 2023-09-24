import os
from tqdm import tqdm
import collections
import argparse
import json

def load_corpus(path='data/corpus.jsonl'):
    data = {}
    n_removed = 0
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            docid = item.pop('doc_id')
            title = item.pop('title', '')
            description = item.pop('description', '')
            data_type = data.pop('type', '')
            asin = data.pop('asin', '')

            if (data_type != 'error'):
                data[str(docid)] = {'title': title, 'description': description}
            else:
                n_removed +=1

    print(f"{n_removed} have been removed")
    return data

def load_predictions(path):
    """
    `id`: string document identifier.
    `contents`: a list of predictied query.
    """
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            docid = item.pop('id')
            data[str(docid)] = " ".join(item['contents'])
    return data

def main(corpus_path, pred_path, output_path, title=False, description=False):
    corpus = load_corpus(corpus_path)
    predictions = load_predictions(pred_path)

    with open(output_path, 'w') as fout:

        for docid in tqdm(corpus):
            contents = ""
            if title:
                contents += corpus[docid]['title']
            if description:
                contents += corpus[docid]['description']

            print(contents)
            fout.write(json.dumps({
                'id': docid, 
                'contents': contents + " " + predictions[docid]
            }, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default='data/corpus.jsonl')
    parser.add_argument("--prediction_jsonl", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_title", default=False, action='store_true')
    parser.add_argument("--use_desc", default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'corpus.jsonl')

    main(args.input_jsonl, args.prediction_jsonl, output_path, title=args.use_title, description=args.use_desc)
    print('done')




