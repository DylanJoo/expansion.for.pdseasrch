mkdir -p runs

prefix=trec-pds-expanded
python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-title.prod2query.trec \
    --index_dir indexing/test \
    --k 1000 --k1 0.5 --b 0.3

