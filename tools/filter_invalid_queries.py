import os
from tqdm import tqdm
import collections
import argparse
import json

def load_query(path='data/qid2query.tsv'):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.split('\t')
            data[str(qid.strip())] = qtext.strip()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    parser.add_argument("--qrels", type=str)
    # output
    parser.add_argument("--qrels_filtered", type=str)
    parser.add_argument("--query_filtered", type=str)
    args = parser.parse_args()

    query = load_query(args.query)

    # remove counter
    filtered_qrels = collections.defaultdict(list)
    valid_queries = []
    fout = open(args.qrels_filtered, 'w')
    with open(args.qrels, 'r') as f:
        for line in tqdm(f):
            qid, _, docid, relevance = line.strip().split('\t')

            query_tokens = query[qid].split(' ')
            if ( len(query_tokens) == 1) and (query_tokens[0].startswith("B")):
                filtered_qrels[qid].append( (docid, relevance) )
            elif ( len(query_tokens) == 1) and (query_tokens[0] == ""):
                filtered_qrels[qid].append( (docid, relevance) )
            else:
                valid_queries.append(qid)
                fout.write(line)

    # filter report 
    print("Filtered query:")
    print([query[qid] for qid in filtered_qrels])

    ## write the new subset of query
    with open(args.query_filtered, 'w') as fout:
        for qid in set(valid_queries):
            fout.write(f"{qid}\t{query[qid]}\n")

    print(f"Number of query filtered: {len(filtered_qrels)}")



