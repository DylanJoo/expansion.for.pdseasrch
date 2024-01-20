import json
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
from datasets import Dataset
from tools import batch_iterator, load_images
from transformers import BlipForQuestionAnswering
from transformers import AutoProcessor
from mlsr_utils import *
from PIL import Image
from tools import init_tokenizer
import torch.nn as nn

def generate_vocab_vector(batch, model, processor, minimum=0, device='cpu', max_length=256, quantization_factor=1000):
    """
    params: batch: Dict[List[str]]
    returns: vectors: List[Dict]
    """
    # prepare text and images
    texts = [f"{t} [SEP] {d}" for t, d in zip(batch['title'], batch['description'])]
    images = []
    image_blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
    for img in batch['image_path']:
        try:
            images.append(Image.open(img).convert('RGB').resize((384, 384)))
        except:
            images.append(image_blank)

    # tokenization
    inputs = processor(
            images=images, 
            text=texts,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length
    ).to(device)

    # use the `encode` function in BlipEncoders
    with torch.no_grad():
        outputs = model.generate(
                **inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=16
        )

        logits = torch.cat([outputs.scores[i][:, None, :] for i in range(len(outputs.scores))], dim=1)
        decoded_token_ids = outputs.sequences[:, 1:] # will not include the firs token in logits
    
    ## get tokens, strings, offset_mapping
    tokenizer = processor.tokenizer
    strings, offset_mapping, attention_mask = batch_transform_token_ids(
            tokenizer, 
            decoded_token_ids,
            return_attention_mask=True
    )
    bow_weights = batch_map_word_values(
            logits, 
            decoded_token_ids, 
            strings, 
            offset_mapping, 
            is_pooled=False
    )

    ## filter the appeared tokens, mask the appeared
    relu = nn.ReLU(inplace=False)
    mask = torch.ones(logits.size(0), 1, logits.size(-1)).to(logits.device)
    mask.scatter_(-1, decoded_token_ids.unsqueeze(1), 0)
    logits = logits * mask

    ## mask the un-attented
    if attention_mask is not None: 
        attention_mask = attention_mask.to(logits.device)
        doc_reps, _ = torch.max(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
    else:
        doc_reps = torch.log(1 + relu(logits))

    ## filter the apperaed tokens (that have been transformed into words)
    cols = torch.nonzero(doc_reps).cpu().numpy()

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    # sort them 
    def sort_dict(dictionary, quantization_factor, bow_dictionary=None):
        # here is the token parts
        d = {k: v*quantization_factor for (k, v) in dictionary if v >= minimum}

        if bow_dictionary is not None:
            d_bows = {w+"@": v*quantization_factor for (w, v) in bow_dictionary if v >= minimum}

        d = {reverse_voc[k]: round(v, 2) for k, v in d.items()}
        d.update(d_bows)

        sorted_d = {k: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        return sorted_d

    return [sort_dict(weight, quantization_factor, bow_weights[i].items()) for i, weight in weights.items()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--img_collection", type=str, default=None)
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--model_name_or_dir", type=str, default=None)
    parser.add_argument("--processor_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # load model and processor
    model = BlipForQuestionAnswering.from_pretrained(args.model_name_or_dir)
    model.to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.processor_name)
    processor = init_tokenizer(processor)
    reverse_voc = {v: k for k, v in processor.tokenizer.vocab.items()}

    # add additional 2 tokens
    offset = model.text_decoder.config.vocab_size - len(reverse_voc)
    for i in range(offset):
        print(f"add {i} offset token: ")
        reverse_voc.update({len(reverse_voc): f"[unused_{i}]"})

    # load data: image
    images = load_images(args.img_collection)

    # load data
    data_list = []
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data = {'doc_id': item['doc_id'],
                    'title': item['title'],
                    'description': item['description'] }
            image = images.get(str(item['doc_id']), None)
            if image:
                data.update({'image_path': image})
            data_list.append(data)

            # remove this after pretesting
            if len(data_list) >= 100 and args.debug:
                break

    dataset = Dataset.from_list(data_list)
    print(dataset)

    # preparing batch 
    vectors = []
    data_iterator = batch_iterator(dataset, args.batch_size, False)
    for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
        batch_vectors = generate_vocab_vector(
                batch=batch,
                model=model,
                processor=processor,
                minimum=args.minimum,
                device=args.device,
                max_length=args.max_length,
                quantization_factor=args.quantization_factor
        )
        assert len(batch['doc_id']) == len(batch_vectors), \
                'Mismatched amount of examples'

        vectors += batch_vectors

    # outout writer
    fout = open(args.collection_output, 'w')

    # collection and re-dump the collections
    for i, example in enumerate(data_list):
        doc_id = example.pop('doc_id')
        title = example.pop('title', "")
        description = example.pop('description', "")

        fout.write(json.dumps({
            "id": doc_id, 
            "contents": title + " " + description,
            "vector": vectors[i]
        }, ensure_ascii=False)+'\n')

    fout.close()

