# t5-base-product2query
PRED=/home/jhju/datasets/EXP4PDS/predictions/
MODEL=ckpt
for ckpt in 10000 15000 20000;do 
    python3 tools/concat_predict_to_corpus.py \
        --input_jsonl data/corpus.jsonl  \
        --prediction_jsonl ${PRED}/corpus.${MODEL}-${ckpt}.pred.jsonl \
        --output_dir data/expanded_corpus/${MODEL}-${ckpt}/ \
        --use_title
done

# baseline
# PRED=/home/jhju/datasets/EXP4PDS/predictions/
# for MODEL in t5-base;do 
#     python3 tools/concat_predict_to_corpus.py \
#         --input_jsonl data/corpus.jsonl  \
#         --prediction_jsonl ${PRED}/corpus.${MODEL}.pred.jsonl \
#         --output_dir data/expanded_corpus/${MODEL}/ \
#         --use_title
# done
