import requests
import torch
from typing import Union
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.processing_utils import ProcessorMixin
from PIL import Image

@dataclass
class Product2Query:
    processor: Union[ProcessorMixin] = None
    template_src: str = "What is the query for the product? title: {0} query: "
    template_tgt: str = "{0}"
    max_src_length: int = 384
    max_tgt_length: int = 16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        images = [Image.open(b['image']).convert('RGB').resize((384, 384)) for b in features]
        texts = [self.template_src.format(b['title']) for b in features]
        labels = [self.template_tgt.format(b['query']) for b in features]

        inputs = self.processor(
                images=images, 
                text=texts,
                max_length=self.max_src_length,
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                padding=True
        )

        targets = self.processor(
                text=labels, 
                max_length=self.max_tgt_length,
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                padding=True
        )
        inputs['labels'] = targets.input_ids
        inputs['decoder_attention_mask'] = targets.attention_mask

        return inputs
