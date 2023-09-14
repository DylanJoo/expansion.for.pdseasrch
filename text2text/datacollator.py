import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class Product2Query:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_src_length: Optional[int] = 128
    max_tgt_length: Optional[int] = 64
    return_tensors: str = "pt"
    template: str = "summarize: title: {0} description: {1}"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        inputs = self.tokenizer(
                [self.template.format(batch['title'], batch['description'])\
                        for batch in features],
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors=self.return_tensors
        )

        targets_ids = self.tokenizer(
                [batch['query'] for batch in features],
                max_length=self.max_tgt_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors=self.return_tensors
        ).input_ids
        targets_ids[targets_ids == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = targets_ids

        return inputs

# @dataclass
# class DataCollatorForDesc2Title:
#     tokenizer: Union[PreTrainedTokenizerBase] = None
#     pad_to_multiple_of: Optional[int] = None
#     padding: Union[bool, str, PaddingStrategy] = True
#     truncation: Union[bool, str] = True
#     max_src_length: Optional[int] = 512
#     max_tgt_length: Optional[int] = 64
#     return_tensors: str = "pt"
#     prefix: str = "summarize: "
#
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#
#         texts_src = []
#         texts_tgt = []
#         for batch in features:
#             description = batch['description']
#             title = batch['title']
#             if (len(description)>5) and (len(title)>0):
#                 texts_src += [description]
#                 texts_tgt += [title]
#
#         inputs = self.tokenizer(
#                 [f"{self.prefix}{src}" for src in texts_src],
#                 max_length=self.max_src_length,
#                 truncation=True,
#                 padding=True,
#                 return_tensors='pt'
#         )
#         target_ids = self.tokenizer(
#                 texts_tgt,
#                 max_length=self.max_tgt_length,
#                 padding=True,
#                 return_tensors='pt'
#         ).input_ids
#         target_ids[target_ids == self.tokenizer.pad_token_id] = -100
#         inputs['labels'] = target_ids
#
#         return inputs
#
