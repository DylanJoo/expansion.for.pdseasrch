# append the document splade vectors
BLIP_129M=DylanJHJ/blip-base-129M
ckpt=25000
MODEL=models/blip-base-prt-mlsr-wmtlm/checkpoint-$ckpt
mkdir -p data/mlsr_corpus/$ckpt

# python multimodal2text/BLIP/append_collection_voc_vectors.py \
#     --collection data/corpus.jsonl \
#     --img_collection data/corpus-images.txt \
#     --collection_output data/mlsr_corpus/corpus_2ep.jsonl \
#     --model_name_or_dir $MODEL \
#     --processor_name $BLIP_129M \
#     --batch_size 2 \
#     --max_length 512 \
#     --device cuda \
#     --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input data/mlsr_corpus/$ckpt \
  --index indexing/trec-pds-simplified-mlsr \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
