mkdir -p indexing
mkdir -p data

# title
python3 text2text/filter_corpus.py \
    --input_jsonl data/corpus.jsonl \
    --output_dir data/simplified_corpus \
    --setting title

# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#       --input data/title_corpus \
#       --index indexing/trec-pds-title/ \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 4

# simplified
python3 text2text/filter_corpus.py \
    --input_jsonl data/corpus.jsonl \
    --output_dir data/simplified_corpus \
    --setting simplified

# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#       --input data/simplified_corpus \
#       --index indexing/trec-pds-simplified/ \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 4

# full
python3 text2text/filter_corpus.py \
    --input_jsonl data/corpus.jsonl \
    --output_dir data/full_corpus \
    --setting full

# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#       --input data/full_corpus \
#       --index indexing/trec-pds-full/ \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 4
