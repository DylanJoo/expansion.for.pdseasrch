import re
# import os
# from tqdm import tqdm
# import collections
# import argparse
# import json
# from tools import load_title, load_run, load_query

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run", type=str)
#     parser.add_argument("--topk", type=int, default=5)
#     parser.add_argument("--truncate", type=int, default=1e10)
#     args = parser.parse_args()
#
#     titles = load_title()
#     runs = load_run(args.run)
#     queries = load_query()
#
#     print(args.run)
#     filename = args.run.replace('run', 'log')
#     fout = open(filename, 'w')
#
#     for i, (qid, doc_list) in enumerate(runs.items()):
#         if i > args.truncate:
#             break
#         query = queries[qid]
#         fout.write(f"* {query}\n")
#         for docid in doc_list[:args.topk]:
#             title = titles.get(docid, f"NA:{docid}")
#             fout.write(f"# {title}\n")
#         fout.write("\n")
#

QRELS='/tmp2/trec/pds/data/qrels/product-search-dev-filtered.qrels'
RUN='/tmp2/trec/pds/data/qrels/pyserini-full-dev.run'
RECALL='recall.1000'
NDCG='ndcg_cut.100'

def run_trec_command(metric):
    import subprocess
    command = f'/tmp2/trec/trec_eval.9.0.4/trec_eval -q -m {metric} {QRELS} {RUN}'
    command = command.split(' ')

    outputs = subprocess.run(command, stdout=subprocess.PIPE)
    outputs = outputs.stdout.decode('utf-8')
    outputs = re.sub('\n', '|', outputs)
    outputs = re.sub('\s+', ' ', outputs)
    outputs = outputs.split('|')

    # collect into per runs
    data = {}
    for output in outputs:
        try:
            metric_, qid, score = output.split()
            if qid.lower() == 'all':
                continue
            else:
                data[qid] = float(score)
        except:
            print(output)

    return data

# start here
judges = run_trec_command(RECALL)
# judges = run_trec_command(NDCG)
sorted_judges = judges.items()
sorted_judges = sorted(sorted_judges, key=lambda x: x[1], reverse=True)
print(sorted_judges)
print([qid for qid, score in sorted_judges if score == 0])
print("total query judged:", len(judges))
