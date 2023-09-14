from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)

@dataclass
class DataCollatorForCrossEncoder:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    padding: Union[bool, str] = True
    separated: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # id pairs
        id_pair_inputs = [(ft['qid'], ft['docid']) for ft in features]

        # text pairs
        if self.separated:
            tokenized_inputs = self.tokenizer(
                [ft['query'] for ft in features],
                [ft['document'] for ft in features],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return tokenized_inputs, id_pair_inputs
        else:
            tokenized_inputs = self.tokenizer(
                [ft['source'] for ft in features],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return tokenized_inputs, id_pair_inputs

@dataclass
class DataCollatorForBiEncoder:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_p_length: Optional[int] = None
    max_q_length: Optional[int] = None
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        tokenized_inputs_source1 = self.tokenizer(
            [ft['source1'] for ft in features],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_q_length,
            return_tensors='pt'
        )
        tokenized_inputs_source2 = self.tokenizer(
            [ft['source2'] for ft in features],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_p_length,
            return_tensors='pt'
        )

        id_pair_inputs = [(ft['qid'], ft['docid']) for ft in features]
        return tokenized_inputs_source1, tokenized_inputs_source2, id_pair_inputs
