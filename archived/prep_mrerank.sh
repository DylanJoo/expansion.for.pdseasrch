python multimodalrerank/convert_qrel_to_seq2seq.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --query data/qid2query.tsv \
    --index_dir indexing/trec-pds-title/ \
    --qrel data/product-search-train.qrels \
    --output data/trec-pds.train.mrerank.hn-v1.jsonl \
    --thres 1

