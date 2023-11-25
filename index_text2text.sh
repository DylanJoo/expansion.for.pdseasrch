for folder in data/expanded_corpus/t5-base-product2query*$1*;do
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input ${folder} \
        --index indexing/trec-pds-expanded-${folder##*/}/ \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4
done

