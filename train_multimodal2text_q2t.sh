export CUDA_VISIBLE_DEVICES=0
VQA=Salesforce/blip-vqa-base
BLIP_129M=DylanJHJ/blip-base-129M
# PDS_PRETRAIN=models/blip-vqa-base-product2title/checkpoint-50000
PDS_PRETRAIN=models/blip-product2title-VE/checkpoint-50000

python3 multimodal2text/BLIP/train_q2t.py \
    --model_name_or_path $PDS_PRETRAIN \
    --config_name $BLIP_129M \
    --processor_name $BLIP_129M \
    --train_file data/trec-pds.train.m2t.product2query.jsonl \
    --max_src_length 16 \
    --max_tgt_length 16 \
    --output_dir models/blip-base-query2title \
    --overwrite_output_dir true \
    --do_train  \
    --max_steps 20000 \
    --save_steps 10000 \
    --save_strategy steps \
    --eval_steps 2000 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --remove_unused_columns false \
    --report_to wandb \
    --overwrite_output_dir true \
    --template_src "title: {0} context: {1}"\
    --template_tgt "{0}" \
    --freeze_text_decoder true \
    --freeze_vision_encoder true \
    --run_name blip-base-q2t-with-pretrain-text-decoder-only
