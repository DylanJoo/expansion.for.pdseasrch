mkdir -p indexes
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/filtered \
  --index indexing/trec-pds-simplified/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4
