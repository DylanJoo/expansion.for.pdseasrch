for ckpt in 20000;do
    MODEL=~/expansion.for.pdseasrch/models_new/t5-base-product2query/checkpoint-${ckpt}
    DIR_OUT=data/expanded_corpus/t5-base-product2query-${ckpt}
    mkdir -p $DIR_OUT
    python text2text/generate.py \
        --collection data/corpus.jsonl \
        --model_name $MODEL \
        --model_hf_name t5-base \
        --do_sample \
        --top_k 10 \
        --batch_size 40 \
        --max_src_length 512 \
        --max_tgt_length 16 \
        --num_return_sequences 10  \
        --output_jsonl $DIR_OUT/corpus.jsonl \
        --template 'summarize: {0} | {1} | {2}' \
        --device cuda
done
