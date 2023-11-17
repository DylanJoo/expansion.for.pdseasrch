export CUDA_VISIBLE_DEVICES=0
python3 multimodal2text/BLIP/train.py \
    --model_name_or_path Salesforce/blip-vqa-base \
    --config_name Salesforce/blip-vqa-base \
    --processor_name Salesforce/blip-vqa-base \
    --train_file data/trec-pds.train.m2t.product2query.jsonl \
    --max_src_length 128 \
    --max_tgt_length 16 \
    --output_dir models/blip-vqa-base-product2query/ \
    --overwrite_output_dir true \
    --do_train --do_eval \
    --save_strategy steps \
    --max_steps 30000 \
    --save_steps 2500 \
    --eval_steps 500 \
    --save_strategy steps \
    --evaluation_strategy steps \
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
    --template_src "{0} What is the possible query for this product?"\
    --template_tgt "{0}" \
    --run_name blip-vqa-base-pds
