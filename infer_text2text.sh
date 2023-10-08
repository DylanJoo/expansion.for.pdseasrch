FILE=data/corpus.jsonl
mkdir -p results/

for ckpt in 10000 15000 20000;do
    MODEL=models/t5-base-product2query/checkpoint-${ckpt}
    FILE_OUT=results/corpus.ckpt-${ckpt}.pred.jsonl
    python text2text/generate.py \
        --collection data/corpus.jsonl \
        --model_name $MODEL \
        --model_hf_name google/t5-v1_1-base \
        --do_sample \
        --top_k 10 \
        --batch_size 40 \
        --max_src_length 512 \
        --max_tgt_length 64 \
        --num_return_sequences 10  \
        --output_jsonl $FILE_OUT \
        --device cuda
done
