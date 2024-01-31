# splade++ baseline
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
# INDEX=indexing/trec-pds-full-lsr
# INDEX=indexing/trec-pds-full-lsr-both

## test
# orig
python3 retrieval/lsr_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-lsr-simplified-orig.run \
    --index indexing/trec-pds-simplified-lsr-orig \
    --batch_size 8 \
    --device cuda:0 \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \

# both
# python3 retrieval/lsr_search.py \
#     --query data/qid2query-test.tsv \
#     --output runs/test-lsr-simplified-both.run \
#     --index indexing/trec-pds-simplified-lsr-both \
#     --batch_size 8 \
#     --device cuda:0 \
#     --use_lexical --include_both \
#     --encoder $MODEL_NAME \
#     --k 1000 --min_idf 0 \

# plus
# python3 retrieval/lsr_search.py \
#     --query data/qid2query-test.tsv \
#     --output runs/test-lsr-simplified-plus.run \
#     --index indexing/trec-pds-simplified-lsr-plus \
#     --batch_size 8 \
#     --device cuda:0 \
#     --use_lexical \
#     --encoder $MODEL_NAME \
#     --k 1000 --min_idf 0 \
