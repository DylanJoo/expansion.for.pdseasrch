# for ckpt in 20000;do
#     DIR_OUT=data/expanded_corpus/blip-vqa-base-product2query-$ckpt
#     MODEL=models/blip-vqa-base-product2query/checkpoint-$ckpt/
#     mkdir -p $DIR_OUT
#     python multimodal2text/BLIP/generate.py \
#         --collection data/corpus.jsonl \
#         --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
#         --model_name $MODEL \
#         --model_hf_name Salesforce/blip-vqa-base \
#         --batch_size 64 \
#         --max_src_length 128 \
#         --max_tgt_length 10 \
#         --num_beams 1 \
#         --num_return_sequences 1 \
#         --output_jsonl $DIR_OUT/corpus.jsonl \
#         --template_src "{0} What is the possible query for this product?"\
#         --device cuda:2
# done

for ckpt in 25000 37500 50000;do
    DIR_OUT=data/expanded_corpus/blip-vqa-base-product2title-$ckpt
    MODEL=models/blip-vqa-base-product2title/checkpoint-$ckpt/
    mkdir -p $DIR_OUT
    python multimodal2text/BLIP/generate.py \
        --collection data/corpus.jsonl \
        --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
        --model_name $MODEL \
        --model_hf_name Salesforce/blip-vqa-base \
        --batch_size 32 \
        --max_src_length 128 \
        --max_tgt_length 10 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --output_jsonl $DIR_OUT/corpus.jsonl \
        --template_src "What is the possible title for this product? title:"\
        --device cuda:2
done
