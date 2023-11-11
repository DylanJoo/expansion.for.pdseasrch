FILE=data/corpus.jsonl
mkdir -p results/

# t5-base # failed 
MODEL=t5-base
FILE_OUT=results/corpus.t5-base.pred.jsonl
python text2text/generate.py \
    --collection data/corpus.jsonl \
    --model_name $MODEL \
    --model_hf_name t5-base \
    --num_beams 4 \
    --batch_size 64 \
    --max_src_length 512 \
    --max_tgt_length 64 \
    --num_return_sequences 1 \
    --output_jsonl $FILE_OUT \
    --template 'summarize: {0} | {1} | {2}' \
    --device cuda:2

# bart-large
# MODEL=facebook/bart-large-cnn
# FILE_OUT=results/corpus.bart-large-cnndm.pred.jsonl
# python text2text/generate.py \
#     --collection data/corpus.jsonl \
#     --model_name $MODEL \
#     --model_hf_name $MODEL \
#     --num_beams 4 \
#     --batch_size 36 \
#     --max_src_length 512 \
#     --max_tgt_length 64 \
#     --num_return_sequences 1  \
#     --output_jsonl $FILE_OUT \
#     --early_stopping  \
#     --template '{0} {1}'\
#     --device cuda:2
