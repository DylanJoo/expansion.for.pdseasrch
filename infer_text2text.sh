FILE=data/corpus.sample.jsonl
MODEL=DylanJHJ/t5-base-product2query

python text2text/generate.py \
    --collection_sim $FILE \
    --model_name $MODEL \
    --tokenizer_name t5-base \
    --do_sample \
    --top_k 10 \
    --batch_size 32 \
    --max_src_length 256 \
    --max_tgt_length 64 \
    --num_return_sequences 10  \
    --output_jsonl ${FILE/jsonl/pred.jsonl}\
    --device cuda:2
