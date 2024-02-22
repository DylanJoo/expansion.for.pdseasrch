import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from pyserini.encode import SpladeQueryEncoder as temp_encoder
from utils import *
import string

def norm(text):
    return text.translate(str.maketrans('', '', string.punctuation))

class SpladeQueryEncoder(temp_encoder):
    def encode(self, text, **kwargs):
        return super().encode(norm(text), **kwargs)

class SpladeQueryLexicalEncoder(SpladeQueryEncoder):
    def __init__(
        self, 
        model_name_or_path, 
        tokenizer_name=None, 
        device='cpu',
        mask_appeared_tokens=False,
        gamma_word=1,
        gamma_token=1
    ):
        super().__init__(model_name_or_path, tokenizer_name, device)
        self.gamma_word = gamma_word
        self.gamma_token = gamma_token
        self.mask_appeared_tokens = mask_appeared_tokens

    def encode(self, text, max_length=256, **kwargs):
        text = norm(text)
        inputs = self.tokenizer([text], 
                                add_special_tokens=True,
                                return_offsets_mapping=True,
                                max_length=max_length, 
                                padding=True,
                                truncation=True, 
                                return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        input_attention = inputs['attention_mask']
        offset_mapping = inputs.pop('offset_mapping')

        batch_logits = self.model(input_ids)['logits']
        batch_aggregated_logits = torch.max(
                torch.log(1 + torch.relu(batch_logits)) * input_attention.unsqueeze(-1), 
                dim=1
        ).values
        batch_aggregated_logits = batch_aggregated_logits.cpu().detach()
        input_ids = input_ids.cpu()
        offset_mapping = offset_mapping.cpu()

        bow_weights = batch_map_word_values(batch_aggregated_logits,
                                            input_ids,
                                            text,
                                            offset_mapping,
                                            is_pooled=True)

        ## get the word-level inferred token-level mask
        if self.mask_appeared_tokens:
            mask = torch.ones(batch_aggregated_logits.size(0), batch_aggregated_logits.size(-1))
            mask.scatter_(-1, input_ids, 0).to(batch_aggregated_logits.device)
            batch_aggregated_logits = (batch_aggregated_logits * mask).numpy()
        else:
            batch_aggregated_logits = batch_aggregated_logits.numpy()

        ## the token-level vectors
        weights = self._output_to_weight_dicts(batch_aggregated_logits)
        for i, weight in enumerate(weights):
            ## [NOTE] more weight on word-level features
            bow_weights_ = {k: v*self.gamma_word for k, v in bow_weights[i].items()}
            weight.update(bow_weights_)

        return self._get_encoded_query_token_wight_dicts(weights)[0]
