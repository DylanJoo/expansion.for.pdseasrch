import json
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
from datasets import Dataset
from tools import batch_iterator, load_images
from encode import BlipForProductEncoder

def generate_vocab_vector(batch, model, minimum=0, device='cpu', max_length=256, quantization_factor=1000):
    """
    params: batch: Dict[List[str]]
    returns: vectors: List[Dict]
    """
    # use the `encode` function in BlipEncoders
    with torch.no_grad():
        doc_reps = model.encode(
                titles=batch['title'],
                descriptions=batch['description'],
                images_path=batch['image_path'],
                max_length=max_length
        )
        # (sparse) doc rep in voc space, shape (30522,)

    # get the number of non-zero dimensions in the rep:
    cols = torch.nonzero(doc_reps).numpy()

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    # sort them 
    def sort_dict(dictionary, quantization_factor):
        d = {k: v*quantization_factor for (k, v) in dictionary if v >= minimum}
        sorted_d = {reverse_voc[k]: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        return sorted_d

    return [sort_dict(weight, quantization_factor) for i, weight in weights.items()]

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
    model = BlipForProductEncoder.from_pretrained(
            args.model_name_or_dir, 
            processor_name=args.processor_name, 
            pooling="max"
    )
    model.to(args.device)
    model.eval()
    reverse_voc = {v: k for k, v in model.processor.tokenizer.vocab.items()}

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

