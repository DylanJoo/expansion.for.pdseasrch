import copy
import torch
import numpy as np
import collections
import torch.nn as nn
import torch.nn.functional as F

def batch_transform_token_ids(tokenizer, token_ids):
    string_to_return = []

    # token_ids --> a token list and a sring
    string_to_return = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # strings --> offset_mapping
    tokenized = tokenizer(
            string_to_return,
            add_special_tokens=False, 
            return_offsets_mapping=True,
            padding=True, 
            return_tensors='pt', 
    ).to(token_ids.device)

    return string_to_return, tokenized.input_ids, tokenized.offset_mapping

def batch_map_word_values(logits, batch_token_ids, strings, offset_mapping, is_pooled=False):

    if is_pooled: 
        # 2 dimension
        weights = logits.gather(1, batch_token_ids).cpu().numpy()
    else:  
        # 3 dimension
        weights = logits.gather(2, batch_token_ids.unsqueeze(2)).squeeze(2).cpu().numpy()

    to_return = []
    for i, (offset, weight) in enumerate(zip(offset_mapping, weights)):
        word_id_to_weights = collections.defaultdict(float)
        words = strings[i]

        start, end = 0, 0
        for j, (start_, end_) in enumerate(offset):
            # retrieve the maximum between tokens
            if end_ != 0:
                w = weight[j]

                if start_ == end:
                    prev = word_id_to_weights[words[start:end]]
                    del word_id_to_weights[words[start:end]]
                    start, end = start, end_
                    word_id_to_weights[words[start: end]] = prev # max(prev, w)

                else: # separated
                    start, end = start_, end_
                    word_id_to_weights[words[start: end]] = w

        word_id_to_weights.pop('[SEP]', None) # remove spec token
        word_id_to_weights.pop('[CLS]', None) # remove spec token
        word_id_to_weights.pop('[PAD]', None) # remove spec token
        to_return.append(word_id_to_weights)
    return to_return

class FLOPS:
    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def splade_max(logits, attention_mask=None, labels_=None):
    relu = nn.ReLU(inplace=False)

    if labels_ is not None:
        mask = torch.ones(logits.size(0), 1, logits.size(-1)).to(logits.device)
        mask.scatter_(-1, labels_.unsqueeze(1), 0)
        logits = logits * mask

    if attention_mask is not None: # [NOTE] masked element in sequence 
        values, _ = torch.max(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
    else:
        values, _ = torch.max(torch.log(1 + relu(logits)), dim=1)
    return values    

def splade_sum(logits, attention_mask=None, labels_=None):
    relu = nn.ReLU(inplace=False)

    if labels_ is not None:
        mask = torch.ones(logits.size(0), 1, logits.size(-1)).to(logits.device)
        mask.scatter_(-1, labels_.unsqueeze(1), 0)
        logits = logits * mask

    if attention_mask is not None: # [NOTE] masked element in sequence 
        values = torch.sum(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
    else:
        values = torch.log(1 + relu(logits))
    return values    

