# text2text zero-shot and text2text fine-tuned
for model in t5-base bart-large-cnndm t5-base-product2query-10000;do
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input data/expanded_corpus/${model} \
        --index indexing/trec-pds-expanded-${model##*/}/ \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4
done
