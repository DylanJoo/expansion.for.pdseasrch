mkdir -p indexing
python -m pyserini.index.lucene \
    --collection JsonCollection \
      --input data/simplified_corpus \
      --index indexing/trec-pds-simplified/ \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4

python -m pyserini.index.lucene \
    --collection JsonCollection \
      --input data/title_corpus \
      --index indexing/trec-pds-title/ \
      --generator DefaultLuceneDocumentGenerator \
      --threads 4
