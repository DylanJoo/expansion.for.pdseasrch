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
    template: str = "What is the query for this product? title: {0}"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        images = [Image.open(requests.get(b['image'], stream=True).raw) \
                for b in features]
        # images = [Image.open(os.path.join(self.image_dir, b['image'])) \
        #         for b in features]
        texts = [self.template_src.format(b['title']) for b in features]
        labels = [self.template_tgt.format(b['query']) for b in features],

        inputs = self.processor(
                images=images, 
                text=text,
                return_tensors=self.return_tensors
        )

        target_ids = self.processor(
                text=labels, 
                return_tensors='pt'
        ).input_ids
        targets_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = targets_ids

        return inputs
