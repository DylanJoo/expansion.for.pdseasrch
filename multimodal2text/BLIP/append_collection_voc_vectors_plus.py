import json
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
from datasets import Dataset
from transformers import AutoProcessor
from tools import batch_iterator, load_images
from mlsr_utils import batch_transform_token_ids, batch_map_word_values
from PIL import Image
from tools import init_tokenizer, batch_iterator
import torch.nn as nn

import string
def norm(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def generate_vocab_vector(batch, model, minimum=0, device='cpu', max_length=256, quantization_factor=1000, mask_appeared_tokens=False):
    """
    params: batch: Dict[List[str]]
    returns: vectors: List[Dict]
    """
    # prepare text and images
    texts = [f"{t} {d}" for t, d in zip(batch['title'], batch['description'])]
    images = []
    image_blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
    for img in batch['image_path']:
        try:
            images.append( Image.open(img).convert('RGB').resize((384, 384)) )
        except:
            images.append(image_blank)

    # tokenization
    inputs = processor(images=images, 
                       text=texts,
                       return_tensors='pt',
                       padding=True,
                       truncation=True,
                       max_length=max_length,
                       return_attention_mask=True)
    inputs['input_ids'][:, 0] = processor.tokenizer.enc_token_id
    B, L = inputs['input_ids'].size(0), 16
    inputs['decoder_input_ids'] = torch.arange(-1, L+2)[:L].repeat((B, 1))
    inputs['decoder_input_ids'][:, 0] = model.decoder_start_token_id
    inputs = inputs.to(device)

    with torch.no_grad():
        # generation
        # outputs = model.generate(**inputs, 
        #                          return_dict_in_generate=True, 
        #                          output_scores=True, 
        #                          max_new_tokens=64)
        # logits = torch.cat([outputs.scores[i][:, None, :] \
        #         for i in range(len(outputs.scores))], dim=1)
        # decoded_token_ids = outputs.sequences[:, 1:] 
        # doc_reps, _ = torch.max(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
        # doc_reps = torch.sum(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
    
        # inference
        outputs = model(**inputs)
        decoded_token_ids = torch.max(outputs.product_logit, dim=-1).indices
        doc_reps = outputs.product_feat

    ## get tokens, strings, offset_mapping
    strings, encoded_token_ids, offset_mapping = \
            batch_transform_token_ids(processor.tokenizer,
                                      decoded_token_ids)

    relu = nn.ReLU(inplace=False)

    ## it can be retrived from pooled logits
    bow_weights = batch_map_word_values(doc_reps,          
                                        encoded_token_ids,
                                        strings, 
                                        offset_mapping, 
                                        is_pooled=True)

    if mask_appeared_tokens:
        mask = torch.ones(doc_reps.size(0), doc_reps.size(-1), device=device)
        mask.scatter_(-1, decoded_token_ids, 0)
        doc_reps = (doc_reps * mask).cpu()

    ### prevent some of the logits being replace by word-level, and thus no weight at all
    doc_reps[:, processor.tokenizer.pad_token_id] = 1e-3 # small logits 
    ### get the number of non-zero dimensions in the rep:
    cols = torch.nonzero(doc_reps)

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    # sort them
    def sort_dict(dictionary, quantization_factor, bow_dictionary=None):
        d = {reverse_voc[k]: v*quantization_factor for (k, v) in dictionary if v >= minimum}

        if bow_dictionary is not None:
            d_bows = {norm(k): v*quantization_factor for (k, v) in bow_dictionary.items() if v >= minimum}
            d.update(d_bows)

        sorted_d = {k: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True) if v >= minimum}
        sorted_d.pop('[SEP]', None) # remove spec token
        sorted_d.pop('[CLS]', None) # remove spec token
        sorted_d.pop('[PAD]', None) # remove spec token
        return sorted_d

    return [sort_dict(weight, quantization_factor, bow_weights[i]) for i, weight in weights.items()]

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
    parser.add_argument("--mask_appeared_tokens", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # load model and processor
    from models_mlsr_wgen import BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(args.model_name_or_dir)
    model.to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.processor_name)
    processor = init_tokenizer(processor)
    reverse_voc = {v: k for k, v in processor.tokenizer.vocab.items()}

    # add additional 2 tokens
    # offset = model.text_decoder.config.vocab_size - len(reverse_voc)
    # for i in range(offset):
    #     print(f"add {i} offset token: ")
    #     reverse_voc.update({len(reverse_voc): f"[unused_{i}]"})

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
            data.update({'image_path': image})
            data_list.append(data)

            if len(data_list) >= 10 and args.debug:
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
                minimum=args.minimum,
                device=args.device,
                max_length=args.max_length,
                quantization_factor=args.quantization_factor,
                mask_appeared_tokens=args.mask_appeared_tokens
        )
        assert len(batch['doc_id']) == len(batch_vectors), \
                f"Mismatched amount of examples, got {len(batch['doc_id'])} and {len(batch_vectors)}"

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
            "contents": title + " | " + description,
            "vector": vectors[i]
        }, ensure_ascii=False)+'\n')

    fout.close()

