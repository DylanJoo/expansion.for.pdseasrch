mkdir -p runs

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/bm25-sim-dev-title.prod2query.trec \
    --index_dir indexing/trec-pds-expanded-title+prod2query \
    --k 1000 --k1 0.9 --b 0.4

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/bm25-sim-dev-title.prod2query_old.trec \
    --index_dir indexing/trec-pds-expanded-title+prod2query_old \
    --k 1000 --k1 0.9 --b 0.4
