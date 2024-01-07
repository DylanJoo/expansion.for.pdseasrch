CORPUS_DIR=data/expanded_corpus/blip-vqa-base-product2query-20000
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $CORPUS_DIR \
    --index indexing/trec-pds-expanded-blip-vqa-base-product2query \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4
