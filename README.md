# expansion.for.pdseasrch

This project is aiming at enhancing product representation using contents in product pages (texts, image and more). 
Retrieval methods will based on the sparse retrieval and learned sparse retrieval.

---
[Docuemnt](https://docs.google.com/document/d/1KxX3rIW7nBVcREkZ5GRUDD2ECPeXSthNUQLnxbSi464/edit?usp=sharing)
[Sheet](https://docs.google.com/spreadsheets/d/1exPfLltGaaf-4Xf3cw4eEhlh8fmouJjoWg4aZWtZDME/edit?usp=sharing)
[Slide](https://docs.google.com/presentation/d/1INviUYGwyGmfDqzhgTnfemRisEd8CJQTWcpVL0pPFXA/edit?usp=sharing)

---
### Dataset
We used the dataset collected by TREC Prodcut Search Track. 
Each files we used are stored at [product-search huggingface](https://huggingface.co/trec-product-search). 
> The files has connect to my datasets directory: `/home/jhju/datasets/`

| Original Files                             | \# Examples |
|:-------------------------------------------|:------------|
| data/corpus.jsonl                          | 1118658     |
| data/qid2query.tsv                         | 30734       |
| data/product-search-dev.qrels              | 169952      |

| Preprocessed Files                         | \# Examples |
|:-------------------------------------------|:------------|
| data/simplified_corpus/corpus.jsonl        | 1080262     |
<<<<<<< HEAD
| data/qid2query-dev-filtered.tsv            | 8940        |
| data/product-search-dev-filtered.qrels     | 169718      |
=======
| data/qid2query-dev-filtered.tsv            | 8941        |
| data/product-search-dev-filtered.qrels     | 169731      |
>>>>>>> e9023b89fba98e70ce842c194795121ebac11314
| trec-pds.train.product2query.jsonl         | 307492      |
    

Note that some of our prepreocessed datasets/files can be found at this [huggingface hub](https://huggingface.co/datasets/DylanJHJ/pds2023/tree/main).

1. simplified_corpus/corpus.jsonl [huggingface_hub] (#) 
A few products' description/title are missing (38396), we only perform indexing on the rest of them.
```
python3 text2text/filter_corpus.py \
    --input_jsonl data/corpus.jsonl \
    --output_jsonl data/simplified_corpus/corpus.jsonl
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

3. Qrels filtered
We remove 13 queries in original [dev-qrels](data/product-search-dev.qrels) that have no particular information needs. 
The filtered [dev-qrels](data/product-search-filtered-dev.qrels) was converted by this command.
```
python3 tools/filter_invalid_queries.py \
    --query data/qid2query.tsv \
    --qrels data/product-search-dev.qrels \
    --qrels_filtered data/product-search-dev-filtered.qrels \
    --query_filtered data/qid2query-dev-filtered.tsv
# Output
169952it [00:00, 463853.96it/s]
Filtered query:
['B07SDGB8XG', '', 'B01LE7U1PG', 'B074M44VZ6', 'B07R5H8QSY', 'B087CZZNDJ', 'B00MEHLYY8', 'B079SHC4SM', 'B086X41FSY', 'B07H2JS63P', 'B004V23YV0', 'B06XXZWR52', 'B00RINP9HG', 'B00HKC17R6']
Number of query filtered: 14
```

### Results
Check the [goolge sheet](https://docs.google.com/spreadsheets/d/1exPfLltGaaf-4Xf3cw4eEhlh8fmouJjoWg4aZWtZDME/edit?usp=sharing)


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
2. Append the predicted texts as a new corpus
We only append the `title` as there is no significant difference between using it with description.
```
python3 tools/concat_predict_to_corpus.py \
    --input_jsonl data/corpus.jsonl  \
    --prediction_jsonl predictions/corpus.pred.jsonl \
    --output_dir data/title+prod2query_old  \ 
    --use_title 
```

The fine-tuned text-to-text model checkpoint.
```
python3 tools/concat_predict_to_corpus.py \
    --input_jsonl data/corpus.jsonl  \
    --prediction_jsonl predictions/corpus.pred.jsonl 
    --output_dir data/title+prod2query_old  \ 
    --use_title 
```
---
### References:

#### Text-based
- [x] Document Expansion by Query Prediction [(Nogueira et al., 2019)](https://arxiv.org/abs/1904.08375)
- [x] SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking [(Formal et al., 2021)](https://arxiv.org/pdf/2308.00415.pdf)
- [ ] Generative Query Reformulation for Effective Adhoc Search [(Wang et al., 2023)](https://arxiv.org/pdf/2308.00415.pdf)
- [x] Leveraging Customer Reviews for E-commerce Query Generation [(Lien et al., 2022)](https://assets.amazon.science/34/e3/a29bde1d44ca9b4252c38a69459c/leveraging-customer-reviews-for-e-commerce-query-generation.pdf)
- [ ] Lexically-Accelerated Dense Retrieval [(Kulkarni et al., 2023)](https://dl.acm.org/doi/pdf/10.1145/3539618.3591715) 

#### Image-based 
- [ ] BLIP
- [ ] GIT 

#### Multimodal
- [x] MSMO: Multimodal Summarization with Multimodal Output [(Zhu et al., 2023)](https://aclanthology.org/D18-1448.pdf)
- [x] Exploiting Pseudo Image Captions for Multimodal Summarization [(Jiang et al., 2023)](https://arxiv.org/pdf/2305.05496.pdf)
- [x] Flava: A foundational language and vision alignment model [(Singh et al., 2022)](https://arxiv.org/abs/2112.04482)
- [x] Kosmos-2: Grounding Multimodal Large Language Models to the World [(Peng et al., 2023)](https://arxiv.org/abs/2306.14824)
- [x] Understanding Guided Image Captioning Performance across Domains [(Ng et al., 2021)](https://arxiv.org/abs/2012.02339)
- [x] Query Generation for Multimodal Documents [(Kim et al., 2021)](https://aclanthology.org/2021.eacl-main.54/)

#### Benchmark datasets
- [x] Retrieval-augmented Image Captioning [(Ramos et al., 2023)](https://arxiv.org/pdf/2302.08268.pdf)
- [x] FAIR-PMD [(dataset)](https://huggingface.co/datasets/facebook/pmd)
- [x] GeneralAI-GRIT [(details)](https://github.com/microsoft/unilm/tree/master/kosmos-2)

#### Others
- [ ] [(OCR toolkit)](https://github.com/PaddlePaddle/PaddleOCR?fbclid=IwAR0ZHQCfhph9HipDFDtaoozOhcNlrOOSQIExywJTsR9M8BTwbX4A3WPcuKY)
