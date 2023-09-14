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

While some of our prepreocessed dataset can be found at this [huggingface hub](https://huggingface.co/datasets/DylanJHJ/pds2023/tree/main)
1. trec-pds.train.product2query.jsonl [huggingface_hub](#)
- Number of examples: 307492 (we randomly pick 3K examples as validation)

### Baselines

### Text-to-text Method
1. Convert [product-search-train.qrels](#) into a jsonl file.
```
python text2text/convert_qrel_to_seq2seq.py \
    --collection data/corpus.jsonl \
    --query data/qid2query.tsv \
    --qrel data/product-search-train.qrels \
    --output data/trec-pds.train.product2query.jsonl
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
