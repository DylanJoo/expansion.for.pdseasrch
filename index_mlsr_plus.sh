export CUDA_VISIBLE_DEVICES=2
VQA=Salesforce/blip-vqa-base
# wgen=-wgen

# pretrained only
TYPE=prt
CKPT=50000
MODEL=models/blip-base-$TYPE-mlsr-plus/checkpoint-${CKPT}
OUTPUT_DIR=data/mlsr_corpus/plus/$TYPE-$CKPT

# fine-tuned only
# TYPE=ft
# CKPT=20000
# MODEL=models/blip-base-$TYPE-mlsr-plus/checkpoint-${CKPT}/
# OUTPUT_DIR=data/mlsr_corpus/plus/$TYPE-$CKPT/

# pretrained then fine-tuned

## the PLUS version
mkdir -p $OUTPUT_DIR
python multimodal2text/BLIP/append_collection_voc_vectors_plus.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output $OUTPUT_DIR/corpus.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $VQA \
    --batch_size 64 \
    --max_length 512 \
    --device cuda \
    --mask_appeared_tokens \
    --quantization_factor 100 

# pyserini indexing with pretokenized impact vectors
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input $OUTPUT_DIR \
  --index indexing/trec-pds-simplified-mlsr-plus-$TYPE \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized

## the BOTH version
mkdir -p ${OUTPUT_DIR/plus/both}
python multimodal2text/BLIP/append_collection_voc_vectors_plus.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output ${OUTPUT_DIR/plus/both}/corpus.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $VQA \
    --batch_size 64 \
    --max_length 512 \
    --device cuda \
    --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${OUTPUT_DIR/plus/both} \
  --index indexing/trec-pds-simplified-mlsr-both-prt \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
