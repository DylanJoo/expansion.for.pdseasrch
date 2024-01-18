model=$1
qrels=data/product-search-test.qrels

echo 'Analyzing' ${model##*/} 
./trec_eval-9.0.7/trec_eval \
    -c -q -m recall.100 \
    ${qrels} ${model} | cut -f2,3  > compare.tsv

run=runs/test-splade-full.run
./trec_eval-9.0.7/trec_eval \
    -c -q -m recall.100 \
    ${qrels} ${run} | cut -f2,3  > baseline.tsv
