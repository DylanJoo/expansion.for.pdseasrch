folder=$1

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${folder} \
    --index indexing/trec-pds-expanded-${folder##*/}/ \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4

