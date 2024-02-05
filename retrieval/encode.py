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
        inputs = self.tokenizer([norm(text)], 
                                max_length=max_length, 
                                padding='longest',
                                truncation=True, 
                                add_special_tokens=True,
                                return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        input_attention = inputs['attention_mask']
        batch_logits = self.model(input_ids)['logits']

        batch_aggregated_logits, _ = torch.max(torch.log(1 + torch.relu(batch_logits))
                                               * input_attention.unsqueeze(-1), dim=1)
        batch_aggregated_logits = batch_aggregated_logits.cpu().detach()
        input_ids = input_ids.cpu()

        ## get the word-level weights
        strings, offset_mapping, _ = batch_transform_token_ids(
                self.tokenizer, input_ids,
        )
        bow_weights = batch_map_word_values(batch_aggregated_logits,
                                            input_ids,
                                            strings,
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

    def _output_to_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v)*self.gamma_token for k, v in zip(list(col), list(weights))}
            to_return.append(d)
        return to_return

    def _get_encoded_query_token_wight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / self.weight_range * self.quant_range)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return
