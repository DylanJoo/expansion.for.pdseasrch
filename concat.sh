# title + prediction
python3 tools/concat_predict_to_corpus.py \
    --input_jsonl data/corpus.jsonl  \
    --prediction_jsonl results/corpus.ckpt-12000.pred.jsonl \
    --output_dir data/expanded_corpus/t5-base-product2query-12000/ \
    --use_title
