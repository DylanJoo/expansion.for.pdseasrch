export CUDA_VISIBLE_DEVICES=2
COLLECTION_SIM=/tmp2/trec/pds/data/collection/collection_sim.jsonl

python3 ../text2text/train.py \
    --model_name_or_path t5-base \
    --config_name t5-base \
    --tokenizer_name t5-base \
    --train_file $COLLECTION_SIM \
    --max_src_length 256  \
    --max_tgt_length 32 \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir /tmp2/trec/pds/models/t5-base-desc2title \
    --eval_steps 2500 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4

