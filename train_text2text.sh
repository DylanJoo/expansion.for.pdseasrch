# export CUDA_VISIBLE_DEVICES=2

python3 ../text2text/train.py \
    --model_name_or_path t5-base \
    --config_name t5-base \
    --tokenizer_name t5-base \
    --train_file data/trec-pds.train.product2query.jsonl \
    --max_src_length 256  \
    --max_tgt_length 32 \
    --output_dir /models/t5-base-product2query \
    --do_train --do_eval \
    --save_strategy steps \
    --max_steps 10000 \
    --save_steps 2500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --template "summarize: title: {0} description: {1}"

