for ckpt in 20000;do
    MODEL=models/blip-vqa-base-product2query/checkpoint-$ckpt
    python multimodal2text/BLIP/qllrank.py \
        --collection data/corpus.jsonl \
        --img_collection /home/jhju/datasets/pdsearch/corpus-images.txt \
        --run runs/test-bm25-title.trec \
        --query data/qid2query.tsv \
        --run_rerank runs/test-bm25-title.qllrank.trec \
        --model_name $MODEL \
        --model_hf_name DylanJHJ/blip-base-129M \
        --batch_size 500 \
        --top_k 1000 \
        --template_src "title: {0} context: {1}" \
        --device cuda:1
done
