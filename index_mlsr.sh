# append the document splade vectors
mkdir -p data/mlsr_corpus/
BLIP_129M=DylanJHJ/blip-base-129M
MODEL=models/blip-base-prt-mtlm/checkpoint-50000
python multimodal2text/BLIP/append_collection_voc_vectors.py \
    --collection data/corpus.jsonl \
    --img_collection data/corpus-images.txt \
    --collection_output data/mlsr_corpus/corpus_p2t.jsonl \
    --model_name_or_dir $MODEL \
    --processor_name $BLIP_129M \
    --batch_size 4 \
    --max_length 512 \
    --device cuda \
    --quantization_factor 100

# pyserini indexing with pretokenized impact vectors
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input data/splade_corpus \
#   --index indexing/trec-pds-full-splade \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --impact --pretokenized
