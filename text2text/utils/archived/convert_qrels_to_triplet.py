from tqdm import tqdm
import random
import collections
import argparse
import requests
import json
from tools import load_query, load_run, load_qrel, load_title, load_collection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_negatives", type=int, default=5)
    parser.add_argument("--with_desc", default=False, action='store_true')
    parser.add_argument("--output", type=str, default='sample.jsonl')
    args = parser.parse_args()

    query = load_query()
    positives, negatives = load_qrel()
    run = load_run()
    if args.with_desc:
        data = load_collection(append='</s>')
    else:
        data = load_title()

    with open(args.output, 'w') as f:
        for qid, docid_list in tqdm(positives.items()):

            # append positive
            example = {'query': query[qid], 'positive': [], 'negative': []}
            for docid in docid_list:
                try:
                    example['positive'].append(data[docid])
                except:
                    print(f"document+ {docid} not found")

            # append negative
            for docid in negatives[qid]:
                try:
                    example['negative'].append(data[docid])
                except:
                    print(f"document- {docid} not found")

            # append hard negative if smaller then n_negatives
            while (len(example['negative']) < args.n_negatives):
                if (qid in run.keys()) and (len(run[qid]) >= 1): 
                    docid = run[qid].pop(0) # start from the first one
                    if docid not in (example['positive'] + example['negative']):
                        try:
                            example['negative'].append(data[docid])
                        except:
                            print(f"document- {docid} not found")
                else:
                    docid = random.choice(list(data.keys()))
                    example['negative'].append(data[docid])

            # drop if no any positive 
            if len(example['positive']) > 0:
                f.write(json.dumps(example)+'\n')


