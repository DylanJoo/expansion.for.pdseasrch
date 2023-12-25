for ckpt in models/blip-product2title-VE/checkpoint-50000 ;do
    echo 'test fine-tuning' $ckpt
    python multimodal2text/BLIP/testing.py \
        --collection data/corpus.jsonl \
        --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
        --model_name $ckpt \
        --model_hf_name Salesforce/blip-vqa-base \
        --batch_size 128 \
        --max_src_length 512 \
        --max_tgt_length 10 \
        --do_sample \
        --top_k 5 \
        --num_return_sequences 10 \
        --do_text_only \
        --template_src "title: {0} context: {1}:" \
        --device cuda:2 > testing
done

for ckpt in checkpoint-10000-freeze-drop checkpoint-10000-freeze-nodrop checkpoint-20000 checkpoint-20000-freeze-drop;do
    echo 'test fine-tuning' $ckpt
    python multimodal2text/BLIP/testing.py \
        --collection data/corpus.jsonl \
        --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
        --model_name models/blip-vqa-base-product2query/$ckpt \
        --model_hf_name Salesforce/blip-vqa-base \
        --batch_size 128 \
        --max_src_length 512 \
        --max_tgt_length 10 \
        --do_sample \
        --top_k 5 \
        --num_return_sequences 10 \
        --do_text_only \
        --template_src "title: {0} context: {1}"\
        --device cuda:2 >> testing
done
