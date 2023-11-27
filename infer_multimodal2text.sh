for ckpt in 20000 17500;do
    DIR_OUT=data/expanded_corpus/blip-vqa-base-product2query-$ckpt
    MODEL=models/blip-vqa-base-product2query/checkpoint-$ckpt/
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
        --template_src "{0} What is the possible query for this product?"\
        --device cuda:1 
done
