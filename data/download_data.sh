mkdir data
# Official product search track at TREC'23
# 1. corpus.jsonl [huggingface hub]
wget https://huggingface.co/datasets/trec-product-search/product-search-corpus/resolve/main/data/jsonl/corpus.jsonl.gz --directory-prefix=data/
gunzip data/corpus.jsonl.gz 
# 2. qid2query.tsv [huggingface hub]
# wget https://huggingface.co/datasets/trec-product-search/product-search-corpus/resolve/main/data/qid2query.tsv --directory-prefix=data/
# 3. product-search-train-qrels.jsonl [huggingface hub]
# wget https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/resolve/main/data/train/product-search-train.qrels.gz --directory-prefix=data/
