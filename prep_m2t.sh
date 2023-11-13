# 307492
python multimodal2text/convert_qrel_to_seq2seq.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-imgs.jsonl \
    --query data/qid2query.tsv \
    --qrel data/product-search-train.qrels \
    --output data/trec-pds.train.m2t.product2query.jsonl \
    --thres 2
