# append the document splade vectors
COLLECTION=data/corpus.jsonl 
COLLECTION_DIR=data/splade_corpus
MODEL_NAME=naver/splade-cocondenser-ensembledistil
python splade/append_collection_voc_vectors.py \
    --collection $COLLECTION \
    --collection_output $COLLECTION_DIR/corpus.jsonl \
    --model_name_or_dir $MODEL_NAME \
    --batch_size 64 \
    --max_length 384 \
    --device cuda:2 \
    --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
INDEX=indexing/trec-pds-full-splade
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input $COLLECTION_DIR \
  --index $INDEX \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
