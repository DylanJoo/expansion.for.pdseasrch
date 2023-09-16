# expansion.for.pdseasrch

This project is aiming at enhancing product representation using contents in product pages (texts, image and more). 
Retrieval methods will based on the sparse retrieval and learned sparse retrieval.

---
### Dataset
We used the dataset collected by TREC Prodcut Search Track. 
Each files we used are stored at [product-search huggingface](https://huggingface.co/trec-product-search). 

We put all the downloaded files in the directory [data](/data).  It will contain
1. corpus.jsonl [huggingface hub](https://huggingface.co/datasets/trec-product-search/product-search-corpus/blob/main/data/jsonl/corpus.jsonl.gz)
2. qid2query.tsv [huggingface hub](https://huggingface.co/datasets/trec-product-search/product-search-corpus/blob/main/data/qid2query.tsv)
3. product-search-train-qrels [huggingface hub](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/blob/main/data/train/product-search-train.qrels.gz)
4. TBA

Some of our prepreocessed datasets/files can be found at this [huggingface hub](https://huggingface.co/datasets/DylanJHJ/pds2023/tree/main).

| Files                               | \# Examples |
|:------------------------------------|:------------|
| data/corpus.jsonl                   | 1118658     |
| data/filtered/corpus.filtered.jsonl | 1080262     |
| trec-pds.train.product2query.jsonl  | 307492      |
    
1. corpus.filtered.jsonl [huggingface_hub] (#) 
A few products' description/title are missing (38396), we only perform indexing on the rest of them.
```
python3 text2text/filter_corpus.py --input_jsonl data/corpus.jsonl --output_jsonl data/filtered/corpus.filtered.jsonl
```

2. trec-pds.train.product2query.jsonl [huggingface_hub](#)
This training files contains 307492 examples (randomly pick 3K examples as validation). We use the train qrels labels [product-search-train.qrels](#) and convert it into seq2se format. You can find the jsonl file [here (huggingface's hub)](#) or run the following script.
```
python text2text/convert_qrel_to_seq2seq.py \
    --collection data/corpus.jsonl \
    --query data/qid2query.tsv \
    --qrel data/product-search-train.qrels \
    --output data/trec-pds.train.product2query.jsonl
```

### Current Results


### Text-to-text Method

1. Fine-tune on the constructed dataset.
```
TRAIN_SEQ2SEQ=data/trec-pds.train.product2query.jsonl
MODEL_PATH=models/t5-base-product2query 

python3 text2text/train.py \
    --model_name_or_path t5-base \
    --config_name t5-base \
    --tokenizer_name t5-base \
    --train_file ${TRAIN_SEQ2SEQ} \
    --max_src_length 384  \
    --max_tgt_length 32 \
    --output_dir ${MODEL_PATH} \
    --do_train --do_eval \
    --save_strategy steps \
    --max_steps 50000 \
    --save_steps 10000 \
    --eval_steps 500 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --optim adafactor \
    --learning_rate 1e-3 \
    --lr_scheduler_type linear \
    --warmup_steps 1000 \
    --remove_unused_columns false \
    --report_to wandb \
    --template "summarize: title: {0} description: {1}"
```

---
### References:

#### Text-based
- [x] Document Expansion by Query Prediction [(Nogueira et al., 2019)](https://arxiv.org/abs/1904.08375)
- [x] SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking [(Formal et al., 2021)](https://arxiv.org/pdf/2308.00415.pdf)
- [ ] Generative Query Reformulation for Effective Adhoc Search [(Wang et al., 2023)](https://arxiv.org/pdf/2308.00415.pdf)

#### Multimodal
- [x] MSMO: Multimodal Summarization with Multimodal Output [(Zhu et al., 2023)](https://aclanthology.org/D18-1448.pdf)
- [x] Exploiting Pseudo Image Captions for Multimodal Summarization [(Jiang et al., 2023)](https://arxiv.org/pdf/2305.05496.pdf)
- [ ] [(OCR toolkit)](https://github.com/PaddlePaddle/PaddleOCR?fbclid=IwAR0ZHQCfhph9HipDFDtaoozOhcNlrOOSQIExywJTsR9M8BTwbX4A3WPcuKY)

#### Others
- [x] Retrieval-augmented Image Captioning [(Ramos et al., 2023)](https://arxiv.org/pdf/2302.08268.pdf)
