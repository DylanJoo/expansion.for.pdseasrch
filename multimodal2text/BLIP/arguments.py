import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    processor_name: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    datacollator: str = field(default='product2query')
    do_lower_case: bool = field(default=True)

@dataclass
class DataArgs:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_src_length: int = field(default=256)
    max_tgt_length: int = field(default=16)
    title_worddrop: float = field(default=0.0)

@dataclass
class TrainArgs(Seq2SeqTrainingArguments):
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=None)
    image_dropout: float = field(default=0.0)
    text_dropout: float = field(default=0.0)
    evaluation_strategy: Optional[str] = field(default='no')
    template_src: Optional[str] = field(default=None)
    template_tgt: Optional[str] = field(default=None)
    predict_with_generate: bool = field(default=True)
    # freeze
    freeze_text_decoder: bool = field(default=False)
