export CUDA_VISIBLE_DEVICES=1
BLIP_129M=DylanJHJ/blip-base-129M

python3 multimodal2text/BLIP/train_mlsr.py \
    --model_name_or_path $BLIP_129M \
    --config_name $BLIP_129M \
    --processor_name $BLIP_129M \
    --train_file data/trec-pds.train.m2t.product2query.jsonl \
    --max_src_length 128 \
    --max_tgt_length 16 \
    --output_dir models/blip-base-mlsr \
    --overwrite_output_dir true \
    --do_train  \
    --max_steps 20000 \
    --save_steps 10000 \
    --save_strategy steps \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --remove_unused_columns false \
    --report_to wandb \
    --overwrite_output_dir true \
    --template_src "title: {0} context: {1}"\
    --template_tgt "{0}" \
    --run_name blip-base-mlsr
