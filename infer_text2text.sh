for ckpt in 10000 12500 15000 17500 20000;do
    MODEL=~/expansion.for.pdseasrch/models/t5-base-product2query/checkpoint-${ckpt}
    DIR_OUT=data/expanded_corpus/t5-base-product2query-${ckpt}
    mkdir -p $DIR_OUT
    python text2text/generate.py \
        --collection data/corpus.jsonl \
        --model_name $MODEL \
        --model_hf_name t5-base \
        --do_sample \
        --top_k 10 \
        --batch_size 32 \
        --max_src_length 512 \
        --max_tgt_length 10 \
        --num_return_sequences 10  \
        --output_jsonl $DIR_OUT/corpus.jsonl \
        --template "summarize: title: {0} contents: {1}" \
        --device cuda
done

