# splade++ baseline
INDEX=indexing/trec-pds-simplified-mlsr

## test
MODEL_NAME=naver/splade-cocondenser-ensembledistil 
python3 retrieval/splade_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-mlsr-simplified.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \
