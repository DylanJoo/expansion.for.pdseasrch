mkdir -p runs

# model=bart-large-cnndm
# python3 retrieval/bm25_search.py \
#     --query data/qid2query-dev-filtered.tsv \
#     --output runs/dev-bm25-title.prod2summary.${model}.trec \
#     --index_dir indexing/trec-pds-expanded-${model} \
#     --k 1000 --k1 0.5 --b 0.3

# model=t5-base
# python3 retrieval/bm25_search.py \
#     --query data/qid2query-dev-filtered.tsv \
#     --output runs/dev-bm25-title.prod2summary.${model}.trec \
#     --index_dir indexing/trec-pds-expanded-${model} \
#     --k 1000 --k1 0.5 --b 0.3

