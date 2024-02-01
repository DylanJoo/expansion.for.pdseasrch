ENCODER='facebook/contriever-msmarco'
INDEX='/tmp2/Kai/indexes/title_desc_contriever/'
QUERY='/tmp2/chiuws/expansion.for.pdseasrch/data/qid2query-dev-filtered.tsv'
 
python /tmp2/chiuws/expansion.for.pdseasrch/dense_retrieval/codes/dense_search.py \
    --output /tmp2/Kai/runs/contriever-msmarco-title_desc-dev.run  \
    --index $INDEX \
    --encoder_name $ENCODER \
    --query $QUERY \
    --device cuda:2 \
    --batch_size 64
