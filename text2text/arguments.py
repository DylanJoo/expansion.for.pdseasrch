import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)

@dataclass
class DataArgs:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_src_length: int = field(default=256)
    max_tgt_length: int = field(default=16)

@dataclass
class TrainArgs(Seq2SeqTrainingArguments):
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=-1)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=2500)
    evaluation_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    learning_rate: Union[float] = field(default=1e-5)
