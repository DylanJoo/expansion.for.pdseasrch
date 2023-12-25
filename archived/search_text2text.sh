mkdir -p runs

prefix=trec-pds-expanded
model=t5-base-product2query
for model in t5-base bart-large-cnndm t5-base-product2query-10000;do
    # dev
    python3 retrieval/bm25_search.py \
        --query data/qid2query-dev-rewritten.tsv \
        --output runs/dev-bm25-title.${model}.trec \
        --index_dir indexing/${prefix}-${model} \
        --k 1000 --k1 0.5 --b 0.3

    # test
    # python3 retrieval/bm25_search.py \
    #     --query data/qid2query-test.tsv \
    #     --output runs/test-bm25-title.${model}.trec \
    #     --index_dir indexing/${prefix}-${model} \
    #     --k 1000 --k1 0.5 --b 0.3
done
