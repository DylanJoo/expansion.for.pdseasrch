# Our text2text baseline includes
# inference, query prediction
# indexing, expanded document reindex 
# search, the standard bm25 search

# 1. inference
mkdir -p data/expanded_corpus/t5-base
mkdir -p data/expanded_corpus/bart-large-cnndm

## t5-base # we found max length=32 works better than 64.
python text2text/generate.py \
    --collection data/corpus.jsonl \
    --model_name t5-base \
    --model_hf_name t5-base \
    --num_beams 1 \
    --batch_size 64 \
    --max_src_length 512 \
    --max_tgt_length 32 \
    --num_return_sequences 1 \
    --output_jsonl data/expanded_corpus/t5-base/corpus.jsonl \
    --template 'summarize: {0} {1}' \
    --device cuda:2

## bart-large
python text2text/generate.py \
    --collection data/corpus.jsonl \
    --model_name facebook/bart-large-cnn \
    --model_hf_name facebook/bart-large-cnn \
    --num_beams 1 \
    --batch_size 32 \
    --max_src_length 512 \
    --max_tgt_length 64 \
    --num_return_sequences 1  \
    --output_jsonl data/expanded_corpus/bart-large-cnndm/corpus.jsonl \
    --template '{0} {1}'\
    --device cuda:2

# 2. index: document indexing
for baseline in t5-base bart-large-cnndm;do
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input ${baseline} \
        --index indexing/trec-pds-expanded-${baseline##*/}/ \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4
done

# 3. search
mkdir -p runs
for baseline in t5-base bart-large-cnndm;do
    python3 retrieval/bm25_search.py \
        --query data/qid2query-dev-filtered.tsv \
        --output runs/dev-bm25-title.prod2summary.$baseline.trec \
        --index_dir indexing/trec-pds-expanded-$baseline \
        --k 1000 --k1 0.5 --b 0.3
