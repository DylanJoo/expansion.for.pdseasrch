mkdir -p runs

prefix=trec-pds-expanded
model=t5-base-product2query
for ckpt in 10000 12500 15000 17500 20000;do
    python3 retrieval/bm25_search.py \
        --query data/qid2query-dev-filtered.tsv \
        --output runs/dev-bm25-title.prod2query.${ckpt}.trec \
        --index_dir indexing/${prefix}-${model}-${ckpt} \
        --k 1000 --k1 0.5 --b 0.3
done
