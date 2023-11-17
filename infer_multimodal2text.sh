DIR_OUT=data/expanded_corpus/blip-vqa-base-product2query
MODEL=Salesforce/blip-vqa-base

mkdir -p $DIR_OUT
python multimodal2text/BLIP/generate.py \
    --collection data/corpus.jsonl \
    --img_collection /home/jhju/datasets/pds.images/corpus-imgs.txt \
    --model_name $MODEL \
    --model_hf_name Salesforce/blip-vqa-base \
    --batch_size 4 \
    --max_src_length 128 \
    --max_tgt_length 16 \
    --do_sample \
    --top_k 10 \
    --num_return_sequences 1  \
    --output_jsonl $DIR_OUT/corpus.jsonl \
    --template_src "title: {0}. What is the query for this product?"\
    --device cuda:2
