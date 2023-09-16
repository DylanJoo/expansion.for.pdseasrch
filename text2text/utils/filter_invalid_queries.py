import os
from tqdm import tqdm
import collections
import argparse
import json
from tools import load_qrel, load_query

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=str)
    parser.add_argument("--query", type=str)
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

            query_text = query[qid].split(' ')
            if ( len(query_text) == 1) and (query_text[0].startswith("B") ):
                filtered_qrels[qid].append( (docid, relevance) )
            else:
                valid_queries.append(qid)
                fout.write(line)

    # filter report 
    print("Filtered query:")
    queries = [query[qid] for qid in filtered_qrels]
    print(queries)
    ## write the new subset query
    with open(args.query_filtered, 'w') as f:
        for qid in set(valid_queries):
            f.write(f"{qid}\t{query[qid]}\n")

    print(f"Number of query filtered: {len(filtered_qrels)}")



