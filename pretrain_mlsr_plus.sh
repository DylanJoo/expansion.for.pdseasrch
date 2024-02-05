export CUDA_VISIBLE_DEVICES=1
VQA=Salesforce/blip-vqa-base

# without generation
python3 multimodal2text/BLIP/pretrain_mlsr.py \
    --model_name_or_path $VQA \
    --config_name $VQA \
    --processor_name $VQA \
    --train_file data/trec-pds.pretrain.m2t.product2query.jsonl \
    --max_src_length 128 \
    --max_tgt_length 16 \
    --output_dir models/blip-base-prt-mlsr-plus/ \
    --overwrite_output_dir true \
    --do_train \
    --save_strategy steps \
    --max_steps 50000 \
    --save_steps 25000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --remove_unused_columns false \
    --report_to wandb \
    --text_generation true \
    --image_dropout 0.1 \
    --text_dropout 0.1 \
    --title_mask_ratio 0.5 \
    --template_src "{0} {1}"\
    --template_tgt "{0}" \
    --pooling sum \
    --run_name prt-mlsr++
