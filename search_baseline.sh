mkdir -p runs

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-title-filtered.trec \
    --index_dir indexing/trec-pds-title \
    --k 1000 --k1 0.5 --b 0.3 &

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-sim-filtered.trec \
    --index_dir indexing/trec-pds-simplified \
    --k 1000 --k1 0.9 --b 0.4 &

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-full-filtered.trec \
    --index_dir indexing/trec-pds-full \
    --k 1000 --k1 4.68 --b 0.87
