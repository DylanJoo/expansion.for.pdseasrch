# for ckpt in 20000;do
#     DIR_OUT=data/expanded_corpus/blip-vqa-base-product2query-$ckpt
#     MODEL=models/blip-vqa-base-product2query/checkpoint-$ckpt
#     mkdir -p $DIR_OUT
#     python multimodal2text/BLIP/generate.py \
#         --collection data/corpus.jsonl \
#         --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
#         --model_name $MODEL \
#         --model_hf_name DylanJHJ/blip-base-129M \
#         --batch_size 64 \
#         --max_src_length 256 \
#         --max_tgt_length 10 \
#         --do_text_only \
#         --num_beams 1 \
#         --num_return_sequences 1 \
#         --output_jsonl $DIR_OUT/corpus.jsonl \
#         --template_src "title: {0} context: {1}" \
#         --device cuda:1
# done

for ckpt in 50000;do
    DIR_OUT=data/expanded_corpus/blip-pretrain-$ckpt
    MODEL=models/blip-product2title-VE/checkpoint-$ckpt
    mkdir -p $DIR_OUT
    python multimodal2text/BLIP/generate.py \
        --collection data/corpus.jsonl \
        --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
        --model_name $MODEL \
        --model_hf_name DylanJHJ/blip-base-129M \
        --batch_size 64 \
        --max_src_length 256 \
        --max_tgt_length 10 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --output_jsonl $DIR_OUT/corpus.jsonl \
        --template_src "title: {0} context: {1}" \
        --device cuda:2
done
