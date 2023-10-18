# title 
for ckpt in 10000 15000;do 
    python3 tools/concat_predict_to_corpus.py \
        --input_jsonl data/corpus.jsonl  \
        --prediction_jsonl results/corpus.t5-base-product2query-${ckpt}.pred.jsonl \
        --output_dir data/expanded_corpus/t5-base-product2query-${ckpt}/ \
        --use_title
done
