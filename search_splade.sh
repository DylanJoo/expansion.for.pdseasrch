# splade++ baseline
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
INDEX=indexing/trec-pds-full-splade
python3 retrieval/splade_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-splade-full.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \

