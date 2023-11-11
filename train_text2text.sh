python3 text2text/train.py \
    --model_name_or_path google/t5-v1_1-base \
    --config_name google/t5-v1_1-base \
    --tokenizer_name google/t5-v1_1-base \
    --train_file data/trec-pds.train.product2query.jsonl \
    --max_src_length 384  \
    --max_tgt_length 16 \
    --output_dir models_new/t5-base-product2query \
    --overwrite_output_dir true \
    --do_train --do_eval \
    --save_strategy steps \
    --max_steps 30000 \
<<<<<<< HEAD
    --save_steps 2500 \
=======
    --save_steps 5000 \
>>>>>>> origin/main
    --eval_steps 500 \
    --warmup_steps 5000 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
<<<<<<< HEAD
    --lr_scheduler_type constant \
    --remove_unused_columns false \
    --report_to wandb \
    --overwrite_output_dir true \
    --template "{0} | {2} | {1}" \
    --run_name pds-t2t-1e4

=======
    --remove_unused_columns false \
    --report_to wandb \
    --overwrite_output_dir true \
    --template "summarize: title: {0} description: {1}"
>>>>>>> origin/main
