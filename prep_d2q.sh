python tools/convert_qrel_to_seq2seq_t2t.py \
    --collection data/corpus.jsonl \
    --query data/qid2query.tsv \
    --qrel data/product-search-train.qrels \
    --output data/trec-pds.train.t2t.product2query.jsonl \
    --thres 2
