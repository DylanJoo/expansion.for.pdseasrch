import os
import torch
from tqdm import tqdm 
import json
import argparse
from PIL import Image
from transformers import AutoConfig
from datasets import Dataset
from tools import batch_iterator, load_images, load_run, load_query
import collections

def inference(model, processor, query, batch, device, template):
    blank = Image.new('RGB', (384, 384), color=(255, 255, 255))

    doc_ids = [b['doc_id'] for b in batch]
    images = []
    texts = []
    for b in batch:
        #images
        try:
            images.append(
                    Image.open(b['image']).convert('RGB').resize((384, 384))
            )
        except:
            images.append(blank)

        # texts
        texts.append(template.format(b['title'], b['description']))

    processed_input = processor(
            images=images, text=texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
    ).to(device)

    labels = processor(
            text=[query]*len(batch),
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
    ).input_ids.to(device)

    # predict probs
    with torch.no_grad():
        # vision
        image_embeds = model.vision_model(
            pixel_values=processed_input['pixel_values'],
            output_attentions=None,
            output_hidden_states=None,
        )[0]
        image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
        )

        # text encode
        q_embeds = model.text_encoder(
            input_ids=processed_input['input_ids'],
            attention_mask=processed_input['attention_mask'],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]

        # text decode
        losses = model.text_decoder(
            input_ids=labels,
            attention_mask=None,
            encoder_hidden_states=q_embeds,
            encoder_attention_mask=processed_input['attention_mask'],
            labels=labels,
            reduction="none",
        ).loss.detach().cpu().flatten().numpy()

    return list(zip(doc_ids, -losses))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str)
    parser.add_argument("--img_collection", type=str)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--run_rerank", type=str, default=None)
    parser.add_argument("--model_name", default='t5-base', type=str)
    parser.add_argument("--model_hf_name", default='t5-base', type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--template_src", default=None, type=str)
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    # load hf 
    from transformers import BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(args.model_name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_hf_name)

    model = model.to(args.device)
    model = model.eval()

    # load data: run
    run_orig = load_run(args.run, top_k=args.top_k)

    # load data: queries
    queries = load_query(args.query)

    # load data: images
    images = load_images(args.img_collection)

    # load data: textd
    products = {}
    with open(args.collection, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            doc_id = str(item['doc_id'])
            products[doc_id] = {'doc_id': doc_id, 'title': item['title'], 'description': item['description']}
            image = images.get(str(item['doc_id']), None)

            if image:
                products[doc_id].update({'image': image})

    # rank
    run_rerank = collections.OrderedDict()
    for qid, ranklist_orig in tqdm(run_orig.items(), total=len(run_orig)):
        query = queries[qid]

        # inference a set of query-passages pairs
        qll = []
        batch_products = []
        for i, doc_id in enumerate(ranklist_orig):
            batch_products.append(products[doc_id])

            if (len(batch_products) >= args.batch_size) or (i == len(ranklist_orig)-1):

                batch_qll = inference(
                        model, processor, 
                        query, batch_products,
                        args.device, args.template_src
                ) # tuple of (docid, qll)
                batch_products.clear()
                qll.extend(batch_qll)

        # sorted by the qll
        run_rerank[qid] = sorted(qll, key=lambda x: x[1], reverse=True)

    # write
    with open(args.run_rerank, 'w') as f:
        for qid in run_rerank:
            ranklist_rerank = run_rerank[qid]

            for i, (doc_id, score) in enumerate(ranklist_rerank):
                f.write("{} Q0 {} {} {} qllrank\n".format(
                    qid, doc_id, i+1, round(score, 4)
                ))

    print("Done")
