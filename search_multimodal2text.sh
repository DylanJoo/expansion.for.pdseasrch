mkdir -p runs

prefix=trec-pds-expanded
model=blip-vqa-base-product2query-17500
python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-title.blip.prod2query.trec \
    --index_dir indexing/${prefix}-${model} \
    --k 1000 --k1 0.5 --b 0.3
