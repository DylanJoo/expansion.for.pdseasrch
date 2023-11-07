splade_sim=/tmp2/trec/pds/indexes/splade-sim 
splade_sim_title=/tmp2/trec/pds/indexes/splade-sim-title
splade_full=/tmp2/trec/pds/indexes/splade-full

QUERY=/tmp2/trec/pds/data/query/2023-test-queries.tsv
# QUERY=/tmp2/trec/pds/data/query/dev-queries.tsv

# python3 ../codes/splade_search.py \
#     --k 1000 \
#     --min_idf 0 \
#     --index $splade_sim_title \
#     --encoder /tmp2/trec/pds/models/splade-pp \
#     --output ../runs/splade-sim-title.dev.run \
#     --query $QUERY

# python3 ../codes/splade_search.py \
#     --k 1000 \
#     --min_idf 0 \
#     --index $splade_sim \
#     --encoder /tmp2/trec/pds/models/splade-pp \
#     --output ../runs/splade-sim.dev.run \
#     --query $QUERY

python3 ../codes/splade_search.py \
    --k 1000 \
    --min_idf 0 \
    --index $splade_full \
    --encoder /tmp2/trec/pds/models/splade-pp \
    --output ../runs/splade-full.test.run \
    --query $QUERY
