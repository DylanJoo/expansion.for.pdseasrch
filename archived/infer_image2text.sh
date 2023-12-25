DIR_OUT=data/expanded_corpus/blip-captioning-base-product2query
MODEL=Salesforce/blip-image-captioning-base \

mkdir -p $DIR_OUT
python multimodal2text/BLIP/generate.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --model_name $MODEL \
    --model_hf_name Salesforce/blip-image-captioning-base \
    --batch_size 4 \
    --max_src_length 128 \
    --max_tgt_length 20 \
    --do_sample \
    --num_return_sequences 1  \
    --output_jsonl $DIR_OUT/corpus.jsonl \
    --device cuda:2
