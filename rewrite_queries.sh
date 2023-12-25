for ckpt in 20000;do
    MODEL=models/blip-base-query2title/checkpoint-$ckpt
    python multimodal2text/BLIP/rewrite.py \
        --model_name $MODEL \
        --model_hf_name DylanJHJ/blip-base-129M \
        --batch_size 64 \
        --max_src_length 32 \
        --max_tgt_length 16 \
        --num_beams 5 \
        --output_tsv data/qid2query-dev-rewritten.tsv \
        --device cuda:0
done
