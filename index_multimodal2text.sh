for ckpt in 20000;do
    DIR_OUT=data/expanded_corpus/blip-vqa-base-product2query-$ckpt
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input $DIR_OUT \
        --index indexing/trec-pds-expanded-blip-vqa-base-product2query-$ckpt \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4
done

for ckpt in 25000;do
    DIR_OUT=data/expanded_corpus/blip-vqa-base-product2title-$ckpt
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input $DIR_OUT \
        --index indexing/trec-pds-expanded-blip-vqa-base-product2title-$ckpt \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4
done
