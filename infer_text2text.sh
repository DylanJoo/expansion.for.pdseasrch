FILE=data/corpus.jsonl
# MODEL=models/t5-base-product2query
# MODEL=models/t5-base-product2query/old
mkdir -p results/

for ckpt in 10000 20000 30000 40000;do
    MODEL=models/t5-base-product2query/checkpoint-${ckpt}
    FILE_OUT=results/corpus.ckpt-${ckpt}.pred.jsonl
    python text2text/generate.py \
        --collection data/corpus.jsonl \
        --model_name $MODEL \
        --tokenizer_name t5-base \
        --do_sample \
        --top_k 10 \
        --batch_size 32 \
        --max_src_length 512 \
        --max_tgt_length 64 \
        --num_return_sequences 10  \
        --output_jsonl $FILE_OUT \
        --device cuda
done
