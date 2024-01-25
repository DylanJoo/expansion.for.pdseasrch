# splade++ baseline
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
INDEX=indexing/trec-pds-full-lsr
INDEX=indexing/trec-pds-full-lsr-both
# INDEX=indexing/trec-pds-full-lsr-plus

## dev
# MODEL_NAME=naver/splade-cocondenser-ensembledistil 
# INDEX=indexing/trec-pds-full-splade
# python3 retrieval/splade_search.py \
#     --query data/qid2query-dev-filtered.tsv \
#     --output runs/dev-splade-full.run \
#     --index $INDEX \
#     --batch_size 8 \
#     --device cuda:0 \
#     --encoder $MODEL_NAME \
#     --k 1000 --min_idf 0 \

## test
python3 retrieval/splade_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-splade-full-new.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --use_lexical \
    --include_both \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \
