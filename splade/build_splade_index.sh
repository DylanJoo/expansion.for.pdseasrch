# COLLECTION_DIR=/tmp2/trec/pds/data/collection/splade/simplified_title/
# INDEX=/tmp2/trec/pds/indexes/splade-sim-title/
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input $COLLECTION_DIR \
#   --index $INDEX \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --impact --pretokenized

# INDEX=/tmp2/trec/pds/indexes/splade-sim/
# COLLECTION_DIR=/tmp2/trec/pds/data/collection/splade/simplified/
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input $COLLECTION_DIR \
#   --index $INDEX \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --impact --pretokenized

INDEX=/tmp2/trec/pds/indexes/splade-full/
COLLECTION_DIR=/tmp2/trec/pds/data/collection/splade/full/
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input $COLLECTION_DIR \
  --index $INDEX \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized
