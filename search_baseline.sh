mkdir -p runs

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/bm25-sim-dev-filtered.trec \
    --index_dir indexing/trec-pds-simplified \
    --k 1000 --k1 0.9 --b 0.4
