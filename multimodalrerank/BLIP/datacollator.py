import random
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
    template_src: str = "{0}"
    template_tgt: str = "{0}"
    max_src_length: int = 384
    max_tgt_length: int = 16
    image_dropout: float = 0.0
    text_dropout: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        images = [Image.open(b['image']).convert('RGB').resize((384, 384)) for b in features]
        texts = [self.template_src.format(b['title'], b['description']) \
                for b in features]

        # 0: text (drop images)
        # 1: text+image (none)
        # -1: image (drop texts)
        drop_labels = random.choices([-1,0,1], k=len(features), 
                weights=(self.text_dropout, self.image_dropout, 1-self.text_dropout-self.image_dropout))

        # random image drop and text drop
        if self.image_dropout > 0:
            blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
            images = [img if lbl!=0 else blank for img, lbl in zip(images, drop_labels)]

        if self.text_dropout > 0:
            blank = self.template_src.format("", "")
            texts = [txt if lbl!=-1 else blank for txt, lbl in zip(texts, drop_labels)] 
        labels = [self.template_tgt.format(b['query']) for b in features]

        inputs = self.processor(
                images=images, 
                text=[f"[ENC] {t} [SEP]" for t in texts],
                max_length=self.max_src_length,
                return_tensors='pt',
                add_special_tokens=False, # we dont want [SEP]
                return_attention_mask=True,
                truncation=True,
                padding=True
        )

        targets = self.processor.tokenizer(
                text=[f"[DEC] {l}" for l in labels], 
                max_length=self.max_tgt_length,
                return_tensors='pt',
                add_special_tokens=False, # we dont want [SEP]
                return_attention_mask=True,
                truncation=True,
                padding=True
        )
        inputs['labels'] = targets.input_ids
        inputs['decoder_attention_mask'] = targets.attention_mask

        # BLIP special token '[ENC] # already done in tokenization
        # inputs['input_ids'][:, 0] = self.processor.tokenizer.enc_token_id
        # BLIP special token '[DEC] # already done in tokenization
        # inputs['labels'][:, 0] = self.processor.tokenizer.bos_token_id
        return inputs

@dataclass
class Product2Title:
    processor: Union[ProcessorMixin] = None
    template_src: str = "{0}"
    template_tgt: str = "{0}"
    max_src_length: int = 384
    max_tgt_length: int = 16
    image_dropout: float = 0.0
    text_dropout: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        images = [Image.open(b['image']).convert('RGB').resize((384, 384)) for b in features]
        texts = [self.template_src.format(b['title_masked'], b['description']) \
                for b in features]

        # 0: text (drop images)
        # 1: text+image (none)
        # -1: image (drop texts)
        drop_labels = random.choices([-1,0,1], k=len(features), 
                weights=(self.text_dropout, self.image_dropout, 1-self.text_dropout-self.image_dropout))
        if self.image_dropout > 0:
            blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
            images = [img if lbl!=0 else blank for img, lbl in zip(images, drop_labels)]

        if self.text_dropout > 0:
            blank = self.template_src.format("", "")
            texts = [txt if lbl!=-1 else blank for txt, lbl in zip(texts, drop_labels)] 
        labels = [self.template_tgt.format(b['title']) for b in features]

        inputs = self.processor(
                images=images, 
                text=[f"[ENC] {t} [SEP]" for t in texts],
                max_length=self.max_src_length,
                return_tensors='pt',
                add_special_tokens=False, # we dont want [SEP]
                return_attention_mask=True,
                truncation=True,
                padding=True
        )

        targets = self.processor(
                text=[f"[DEC] {l}" for l in labels], 
                max_length=self.max_tgt_length,
                return_tensors='pt',
                add_special_tokens=False, # we dont want [SEP]
                return_attention_mask=True,
                truncation=True,
                padding=True
        )
        inputs['labels'] = targets.input_ids
        inputs['decoder_attention_mask'] = targets.attention_mask

        # BLIP special token '[ENC]
        # inputs['input_ids'][:, 0] = self.processor.tokenizer.enc_token_id
        # BLIP special token '[DEC]
        # inputs['labels'][:, 0] = self.processor.tokenizer.bos_token_id
        return inputs

@dataclass
class ProductRerank:
    processor: Union[ProcessorMixin] = None
    template_src: str = "{0}"
    template_tgt: str = None
    max_src_length: int = 16
    max_tgt_length: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        texts = ["[ENC] {0} [SEP] {1} [SEP]".format(b['query'], b['title']) for b in features]

        inputs = self.processor(
                text=texts,
                max_length=self.max_src_length,
                return_tensors='pt',
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=True,
                padding=True
        )

        return inputs
