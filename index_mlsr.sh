export CUDA_VISIBLE_DEVICES=1,2
VQA=Salesforce/blip-vqa-base
ckpt=25000
pooling=max
wgen=-wgen
MODEL=models/blip-base-prt-mlsr-${pooling}${wgen}/checkpoint-$ckpt/

mkdir -p data/mlsr_corpus/prt-$pooling${wgen}-$ckpt/

python multimodal2text/BLIP/append_collection_voc_vectors.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output data/mlsr_corpus/prt-$pooling${wgen}-$ckpt/corpus.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $VQA \
    --batch_size 64 \
    --max_length 512 \
    --device cuda \
    --quantization_factor 100

# # pyserini indexing with pretokenized impact vectors
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input data/mlsr_corpus/$pooling-$ckpt \
#   --index indexing/trec-pds-simplified-mlsr \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --impact --pretokenized
