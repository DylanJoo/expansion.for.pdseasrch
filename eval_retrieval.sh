model=$1

for run in runs/*$1*;do
    if [[ "$run" == *"dev"* ]]; then
        qrels=data/product-search-dev-filtered.qrels
    fi
    if [[ "$run" == *"test"* ]]; then
        qrels=data/2023-product-qrels.txt
    fi
    echo ${run##*/}
    ./trec_eval-9.0.7/trec_eval \
        -c -m recall.10,100,1000 -m ndcg_cut.5,10,100 \
        ${qrels} ${run} | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
done
