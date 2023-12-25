# Our baseline includes
# preprocessing, filter the corpus into (1) title and (2) simplified
# indexing, using title and title+description
# search, the standard bm25 search
mkdir -p indexing


# # preprocessing [already done]
# ## title
# python3 text2text/filter_corpus.py \
#     --input_jsonl data/corpus.jsonl \
#     --output_dir data/title_corpus \
#     --setting title
#
# ## simplified
# python3 text2text/filter_corpus.py \
#     --input_jsonl data/corpus.jsonl \
#     --output_dir data/simplified_corpus \
#     --setting simplified
# ## full
# python3 text2text/filter_corpus.py \
#     --input_jsonl data/corpus.jsonl \
#     --output_dir data/full_corpus \
#     --setting full

# index: document indexing
for baseline in title simplified full;do
    python -m pyserini.index.lucene \
        --collection JsonCollection \
          --input data/${baseline}_corpus \
          --index indexing/trec-pds-${baseline}/ \
          --generator DefaultLuceneDocumentGenerator \
          --threads 4

# search
mkdir -p runs
python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-title.trec \
    --index_dir indexing/trec-pds-title \
    --k 1000 --k1 0.5 --b 0.3 &

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-sim.trec \
    --index_dir indexing/trec-pds-simplified \
    --k 1000 --k1 0.9 --b 0.4 &

python3 retrieval/bm25_search.py \
    --query data/qid2query-dev-filtered.tsv \
    --output runs/dev-bm25-full.trec \
    --index_dir indexing/trec-pds-full \
    --k 1000 --k1 4.68 --b 0.87
