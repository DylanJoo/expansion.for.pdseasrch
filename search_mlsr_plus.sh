# splade++ baseline
INDEX=indexing/trec-pds-simplified-mlsr-plus-ft

## test
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
python3 retrieval/mlsr_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-mlsr-simplified-plus-ft.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --use_lexical \
    --include_both \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \
