python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input data/expanded_corpus/blip-vqa-base-product2query/ \
    --index indexing/trec-pds-expanded-blip-vqa-base-product2query \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4
