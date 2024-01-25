import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from pyserini.encode import QueryEncoder

class SpladeQueryLexicalEncoder(QueryEncoder):
    def __init__(self, model_name_or_path, tokenizer_name=None, device='cpu'):
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.weight_range = 5
        self.quant_range = 256

    def encode(self, text, max_length=256, **kwargs):
        inputs = self.tokenizer([text], max_length=max_length, padding='longest',
                                truncation=True, add_special_tokens=True,
                                return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        input_attention = inputs['attention_mask']
        batch_logits = self.model(input_ids)['logits']

        batch_aggregated_logits, _ = torch.max(torch.log(1 + torch.relu(batch_logits))
                                               * input_attention.unsqueeze(-1), dim=1)

        ## filter the tokens that can be aggregated into words
        word_mask = selfinputs['input_ids'])
        batch_aggregated_logits = batch_aggregated_logits * word_mask

        batch_aggregated_logits = batch_aggregated_logits.cpu().detach().numpy()

        ## the token-level vectors
        weights = self._output_to_weight_dicts(batch_aggregated_logits)

        ## the word-level vectors
        strings, offset_mapping, _ = batch_transform_token_ids(self.tokenizer, inputs['input_ids'])
        bow_weights = batch_map_word_values(
                batch_aggregated_logits, 
                inputs['input_ids'], 
                strings, 
                offset_mapping, 
                is_pooled=True
        )
        for i, weight in enumerate(weights):
            weight.update(bow_weights[i])

        return self._get_encoded_query_token_wight_dicts(weights)[0]

    def _output_to_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
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
