# append the document splade vectors
TYPE=both
TYPE=plus
TYPE=orig

mkdir -p data/lsr_corpus/$TYPE/
MODEL_NAME=naver/splade-cocondenser-ensembledistil
# # python splade/append_collection_voc_vectors_plus.py \
#     --collection data/simplified_corpus/corpus.jsonl \
#     --collection_output data/lsr_corpus/$TYPE/corpus.jsonl \
#     --model_name_or_dir $MODEL_NAME \
#     --batch_size 64 \
#     --max_length 512 \
#     --mask_appeared_tokens \
#     --device cuda:2 \
#     --quantization_factor 100

python splade/append_collection_voc_vectors.py \
    --collection data/simplified_corpus/corpus.jsonl \
    --collection_output data/lsr_corpus/$TYPE/corpus.jsonl \
    --model_name_or_dir $MODEL_NAME \
    --batch_size 64 \
    --max_length 512 \
    --device cuda:0 \
    --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input data/lsr_corpus/$TYPE \
  --index indexing/trec-pds-simplified-lsr-$TYPE \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
