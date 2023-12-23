model=$1
qrels=data/product-search-dev-filtered.qrels

echo 'Analyzing' ${model##*/} 'and t5-base comparison'
./trec_eval-9.0.7/trec_eval \
    -c -q -m ndcg_cut.100 \
    ${qrels} ${model} | cut -f2,3  > model.tsv

run=runs/dev-bm25-sim.prod2query.trec
./trec_eval-9.0.7/trec_eval \
    -c -q -m ndcg_cut.100 \
    ${qrels} ${run} | cut -f2,3  > t5-base.tsv

# ['36625', '78961']
# df = pd.read_csv('blip.r100.tsv', delimiter='\t', header=None, index_col=[0])
# df1 = pd.read_csv('model.tsv', delimiter='\t', header=None, index_col=[0], names=['compare'])
# df2 = pd.read_csv('t5-base.tsv', delimiter='\t', header=None, index_col=[0], names=['t5-base'])
#
# df = pd.concat([df, df2])
