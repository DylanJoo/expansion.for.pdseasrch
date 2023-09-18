FILE=data/corpus.jsonl
FILE_OUT=${FILE/data/results}
MODEL=models/t5-base-product2query/t5-base-product2query/old
MODEL=models/t5-base-product2query/t5-base-product2query
mkdir -p results/

python text2text/generate.py \
    --collection $FILE \
    --model_name $MODEL \
    --tokenizer_name t5-base \
    --do_sample \
    --top_k 10 \
    --batch_size 40 \
    --max_src_length 512 \
    --max_tgt_length 64 \
    --num_return_sequences 10  \
    --output_jsonl ${FILE_OUT/jsonl/pred.jsonl}\
    --device cuda
