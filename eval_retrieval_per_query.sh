model1=$1
qrels=data/product-search-dev-filtered.qrels

for run in runs/*$1*;do
    echo ${run##*/}
    ./trec_eval-9.0.7/trec_eval \
        -c -q -m recall.100 \
        ${qrels} ${run} | cut -f2,3  > ${model1}.r100.tsv
done

run=runs/dev-bm25-sim.prod2query.trec
./trec_eval-9.0.7/trec_eval \
    -c -q -m recall.100 \
    ${qrels} ${run} | cut -f2,3  > t5-base.r100.tsv

# ['36625', '78961']
# df = pd.read_csv('blip.r100.tsv', delimiter='\t', header=None, index_col=[0])
