# splade++ baseline
## dev
VQA=Salesforce/blip-vqa-base
INDEX=indexing/trec-pds-simplified-mlsr

ckpt=25000
MODEL=models/blip-base-prt-mlsr-wgen-max/checkpoint-$ckpt
# python3 retrieval/splade_search.py \
#     --query data/qid2query-dev-filtered.tsv \
#     --output runs/dev-splade-full.run \
#     --index $INDEX \
#     --batch_size 8 \
#     --device cuda:0 \
#     --encoder $MODEL \
#     --k 1000 --min_idf 0 \

## test
# ckpt=25000
# python3 multimodal2text/BLIP/splade_search.py \
#     --query data/qid2query-test.tsv \
#     --output runs/test-mlsr-simplified.run \
#     --index $INDEX \
#     --batch_size 8 \
#     --device cuda:0 \
#     --encoder $MODEL \
#     --processor $VQA \
#     --k 1000 --min_idf 0 

MODEL_NAME=naver/splade-cocondenser-ensembledistil 
python3 retrieval/splade_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-mlsr-simplified.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --encoder $MODEL_NAME \
    --k 1000 --min_idf 0 \
