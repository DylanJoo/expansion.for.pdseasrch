mkdir -p runs

prefix=trec-pds-expanded
model=t5-base-product2query
<<<<<<< HEAD
model=ckpt
for ckpt in 10000 15000 20000;do
=======
for ckpt in 10000 15000 20000 25000 30000;do
>>>>>>> text-based
    python3 retrieval/bm25_search.py \
        --query data/qid2query-dev-filtered.tsv \
        --output runs/dev-bm25-title.prod2query.${ckpt}.trec \
        --index_dir indexing/${prefix}-${model}-${ckpt} \
        --k 1000 --k1 0.5 --b 0.3
done

