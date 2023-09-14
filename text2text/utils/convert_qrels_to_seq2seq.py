from tqdm import tqdm
import collections
import argparse
import json
from tools import load_query, load_run, load_qrel, load_title, load_collection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--output", type=str, default='sample.jsonl')
    args = parser.parse_args()

    query = load_query()
    positives, negatives = load_qrel()
    run = load_run()
    data = load_collection(append=False)

    with open(args.output, 'w') as f:
        for qid, docid_list in tqdm(positives.items()):

            # append positive
            example = {'target': query[qid], 'source': []}
            for docid in docid_list:
                try:
                    title = data[docid]['title']
                    description = data[docid]['description']
                    example['source'] = f"summarize: title: {title} description: {description}"
                    f.write(json.dumps(example, ensure_ascii=False)+'\n')
                except:
                    print("Missing doc", docid)



