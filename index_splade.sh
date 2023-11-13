# append the document splade vectors
mkdir -p data/splade_corpus/
# MODEL_NAME=naver/splade-cocondenser-ensembledistil
# python splade/append_collection_voc_vectors.py \
#     --collection data/full_corpus/corpus.jsonl \
#     --collection_output data/splade_corpus/corpus.jsonl \
#     --model_name_or_dir $MODEL_NAME \
#     --batch_size 128 \
#     --max_length 512 \
#     --device cuda \
#     --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input data/splade_corpus \
  --index indexing/trec-pds-full-splade \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
