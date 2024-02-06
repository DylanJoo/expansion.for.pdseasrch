# splade++ baseline
INDEX=indexing/trec-pds-simplified-mlsr-both-prt

## test
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
python3 retrieval/lsr_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-mlsr-simplified-both-ft.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:2 \
    --use_lexical \
    --include_both \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \
