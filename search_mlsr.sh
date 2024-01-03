# splade++ baseline
## dev
ckpt=25000
MODEL=models/blip-base-prt-mlsr-wmtlm/checkpoint-$ckpt
BLIP_129M=DylanJHJ/blip-base-129M
INDEX=indexing/trec-pds-simplified-mlsr
# python3 retrieval/splade_search.py \
#     --query data/qid2query-dev-filtered.tsv \
#     --output runs/dev-splade-full.run \
#     --index $INDEX \
#     --batch_size 8 \
#     --device cuda:0 \
#     --encoder $MODEL \
#     --k 1000 --min_idf 0 \

## test
ckpt=25000
MODEL=models/blip-base-prt-mlsr-wmtlm/checkpoint-$ckpt
BLIP_129M=DylanJHJ/blip-base-129M
INDEX=indexing/trec-pds-simplified-mlsr
python3 multimodal2text/BLIP/splade_search.py \
    --query data/qid2query-test.tsv \
    --output runs/test-mlsr-simplified.run \
    --index $INDEX \
    --batch_size 8 \
    --device cuda:0 \
    --encoder $MODEL \
    --processor $BLIP_129M \
    --k 1000 --min_idf 0 \
