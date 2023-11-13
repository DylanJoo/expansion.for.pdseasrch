import requests
import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.processing_utils import ProcessorMixin
from PIL import Image

@dataclass
class Product2Query:
    processor: Union[ProcessorMixin] = None
    image_dir: str = '/home/jhju/datasets/pdsearch/images/'
    template_src: str = "What is the query for the product? title: {0} query: "
    template_tgt: str = "{0}"
    max_src_length: int = 384
    max_tgt_length: int = 16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # images = [Image.open(requests.get(b['image'], stream=True).raw) \
        #         for b in features]
        # texts = [self.template_src.format(b['title']) for b in features]
        # labels = [self.template_tgt.format(b['query']) for b in features],

        images, texts= [], []
        labels = []
        for batch in features:
            try:
                image = Image.open(requests.get(batch['image'], stream=True).raw)
                # image = Image.open(requests.get(batch['image'], stream=True).raw)
                # image = Image.open(os.path.join(self.image_dir, b['image']))
            except:
                image = None

            if image is not None:
                images += [image]
                texts += [self.template_src.format(batch['title'])]
                labels += [batch['query']]

        inputs = self.processor(
                images=images, 
                text=texts,
                max_length=self.max_src_length,
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                padding=True
        )

        target_ids = self.processor(
                text=labels, 
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                padding=True
        ).input_ids
        inputs['labels'] = target_ids

        return inputs
