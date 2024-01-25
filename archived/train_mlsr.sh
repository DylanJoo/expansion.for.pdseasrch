export CUDA_VISIBLE_DEVICES=1
VQA=Salesforce/blip-vqa-base
PRETRAINED=models/blip-base-prt-mlsr-max/checkpoint-25000
PRETRAINED=Salesforce/blip-vqa-base

python3 multimodal2text/BLIP/train_mlsr.py \
    --model_name_or_path $PRETRAINED \
    --config_name $VQA \
    --processor_name $VQA \
    --train_file data/trec-pds.train.m2t.product2query.jsonl \
    --max_src_length 128 \
    --max_tgt_length 16 \
    --output_dir models/blip-base-ft-mlsr \
    --overwrite_output_dir true \
    --do_train \
    --save_strategy steps \
    --max_steps 20000 \
    --save_steps 10000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --remove_unused_columns false \
    --report_to wandb \
    --text_generation false \
    --template_src "{0} [SEP] {1}"\
    --template_tgt "{0}" \
    --pooling max \
    --run_name blip-base-ft+prt-mlsr-max
