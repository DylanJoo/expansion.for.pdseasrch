COLLECTION_SIM=/tmp2/trec/pds/data/predictions/collection_sim_remained.jsonl
PREDICTION=/tmp2/trec/pds/data/predictions/product2query.predicted.remained.jsonl

MODEL=t5-base # zeroshot
MODEL=/tmp2/trec/pds/models/t5-base-product2query/checkpoint-12000 

python ../codes/t5_seq2seq_generation.py \
    --collection_sim $COLLECTION_SIM \
    --model_name $MODEL \
    --tokenizer_name t5-base \
    --top_k 10 \
    --do_sample \
    --batch_size 32 \
    --max_src_length 256 \
    --max_tgt_length 64 \
    --num_return_sequences 10  \
    --output_jsonl $PREDICTION \
    --device cuda:2
