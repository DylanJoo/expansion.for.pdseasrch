MODEL=t5-base
DIR_OUT=data/expanded_corpus/t5-base
mkdir -p $DIR_OUT

python text2text/generate.py \
    --collection data/corpus.jsonl \
    --model_name $MODEL \
    --model_hf_name t5-base \
    --num_beams 1 \
    --batch_size 64 \
    --max_src_length 512 \
    --max_tgt_length 32 \
    --num_return_sequences 1 \
    --output_jsonl $DIR_OUT/corpus.jsonl \
    --template 'summarize: {0} {1}' \
    --device cuda:2

# bart-large
# MODEL=facebook/bart-large-cnn
# DIR_OUT=data/expanded_corpus/bart-large-cnndm
# mkdir -p $DIR
# python text2text/generate.py \
#     --collection data/corpus.jsonl \
#     --model_name $MODEL \
#     --model_hf_name $MODEL \
#     --num_beams 1 \
#     --batch_size 32 \
#     --max_src_length 512 \
#     --max_tgt_length 64 \
#     --num_return_sequences 1  \
#     --output_jsonl $DIR_OUT/corpus.jsonl \
#     --template '{0} {1}'\
#     --device cuda:2
