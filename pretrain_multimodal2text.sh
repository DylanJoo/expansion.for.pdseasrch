export CUDA_VISIBLE_DEVICES=2
BLIP_129M=DylanJHJ/blip-base-129M

# mlsr only
# python3 multimodal2text/BLIP/pretrain_mlsr.py \
#     --model_name_or_path $BLIP_129M \
#     --config_name $BLIP_129M \
#     --processor_name $BLIP_129M \
#     --train_file data/trec-pds.pretrain.m2t.product2query.jsonl \
#     --max_src_length 256 \
#     --max_tgt_length 16 \
#     --output_dir models/blip-base-prt-mlsr/ \
#     --overwrite_output_dir true \
#     --do_train \
#     --save_strategy steps \
#     --max_steps 50000 \
#     --save_steps 25000 \
#     --save_strategy steps \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0.05 \
#     --remove_unused_columns false \
#     --report_to wandb \
#     --overwrite_output_dir true \
#     --image_dropout 0.1 \
#     --text_dropout 0.1 \
#     --title_mask_ratio 0.5 \
#     --template_src "title: {0} context: {1}"\
#     --template_tgt "query: {0}" \
#     --run_name blip-pretrain-mlsr

# mlsr + mtlm
python3 multimodal2text/BLIP/pretrain_mlsr_wmtlm.py \
    --model_name_or_path $BLIP_129M \
    --config_name $BLIP_129M \
    --processor_name $BLIP_129M \
    --train_file data/trec-pds.pretrain.m2t.product2query.jsonl \
    --max_src_length 256 \
    --max_tgt_length 16 \
    --output_dir models/blip-base-prt-mlsr-wmtlm/ \
    --overwrite_output_dir true \
    --do_train \
    --save_strategy steps \
    --max_steps 50000 \
    --save_steps 25000 \
    --save_strategy steps \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --remove_unused_columns false \
    --report_to wandb \
    --overwrite_output_dir true \
    --image_dropout 0.1 \
    --text_dropout 0.1 \
    --title_mask_ratio 0.5 \
    --template_src "title: {0} context: {1}"\
    --template_tgt "query: {0}" \
    --run_name blip-pretrain-mlsr-wtlm

# multiply: the logit of text-only and text-image will first multiply before language modeling
