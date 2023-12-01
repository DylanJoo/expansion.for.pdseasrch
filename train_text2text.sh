python3 text2text/train.py \
    --model_name_or_path t5-base \
    --config_name t5-base \
    --tokenizer_name t5-base \
    --train_file data/trec-pds.train.t2t.product2query.jsonl \
    --max_src_length 384 \
    --max_tgt_length 16 \
    --output_dir models/t5-base-product2query/ \
    --overwrite_output_dir true \
    --do_train --do_eval \
    --max_steps 30000 \
    --save_steps 25000 \
    --eval_steps 500 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --remove_unused_columns false \
    --optim adafactor \
    --report_to wandb \
    --overwrite_output_dir true \
    --template "summarize: title: {0} contents: {1}" \
    --run_name latest-one-t2t
