MODEL_NAME=/tmp2/trec/pds/models/splade-pp

# COLLECTION=/tmp2/trec/pds/data/collection/lucene/simplified_title/doc00.jsonl
# COLLECTION_OUT=/tmp2/trec/pds/data/collection/splade/simplified_title/doc00.jsonl
# python ../codes/impact_encode/append_collection_voc_vectors.py \
#     --collection $COLLECTION \
#     --collection_output $COLLECTION_OUT \
#     --model_name_or_dir $MODEL_NAME \
#     --batch_size 64 \
#     --max_length 32 \
#     --device cuda:2 \
#     --quantization_factor 100 &

# COLLECTION=/tmp2/trec/pds/data/collection/lucene/simplified/doc00.jsonl
# COLLECTION_OUT=/tmp2/trec/pds/data/collection/splade/simplified/doc00.jsonl
# python ../codes/impact_encode/append_collection_voc_vectors.py \
#     --collection $COLLECTION \
#     --collection_output $COLLECTION_OUT \
#     --model_name_or_dir $MODEL_NAME \
#     --batch_size 64 \
#     --device cuda:2 \
#     --quantization_factor 100

COLLECTION=/tmp2/trec/pds/data/collection/lucene/full/doc00.jsonl
COLLECTION_OUT=/tmp2/trec/pds/data/collection/splade/full/doc00.jsonl
python ../codes/impact_encode/append_collection_voc_vectors.py \
    --collection $COLLECTION \
    --collection_output $COLLECTION_OUT \
    --model_name_or_dir $MODEL_NAME \
    --batch_size 64 \
    --max_length 384 \
    --device cuda:2 \
    --quantization_factor 100
