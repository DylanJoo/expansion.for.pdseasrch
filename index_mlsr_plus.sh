export CUDA_VISIBLE_DEVICES=1
VQA=Salesforce/blip-vqa-base
POOLING=max
# wgen=-wgen

# pretrained only
CKPT=50000
MODEL=models/blip-base-prt-mlsr-plus/checkpoint-${CKPT}
OUTPUT_DIR=data/mlsr_corpus/plus/prt-$CKPT

# fine-tuned only
# CKPT=20000
# MODEL=models/blip-base-ft-mlsr-plus/checkpoint-${CKPT}/
# OUTPUT_DIR=data/mlsr_corpus/plus/ft--$CKPT/
# mkdir -p $OUTPUT_DIR

# # pretrained then fine-tuned
# CKPT=20000
# MODEL=models/blip-base-ft+prt-mlsr-${POOLING}${wgen}/checkpoint-${CKPT}/
# OUTPUT_DIR=data/mlsr_corpus/ft+prt-${POOLING}${wgen}-$CKPT/
# mkdir -p $OUTPUT_DIR

mkdir -p $OUTPUT_DIR
python multimodal2text/BLIP/append_collection_voc_vectors_plus.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output $OUTPUT_DIR/corpus.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $VQA \
    --batch_size 32 \
    --max_length 256 \
    --device cuda \
    --mask_appeared_tokens \
    --quantization_factor 100 

mkdir -p ${OUTPUT_DIR/plus/both}
python multimodal2text/BLIP/append_collection_voc_vectors_plus.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output ${OUTPUT_DIR/plus/both}/corpus.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $VQA \
    --batch_size 32 \
    --max_length 256 \
    --device cuda \
    --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input $OUTPUT_DIR \
#   --index indexing/trec-pds-simplified-mlsr \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --impact --pretokenized
