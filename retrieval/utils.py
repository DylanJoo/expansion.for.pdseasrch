import torch
import numpy as np
import collections

# def batch_transform_token_ids(tokenizer, token_ids, return_attention_mask=False):
#     tokens_to_add = []
#     string_to_return = []
#
#     # token_ids --> a token list and a sring
#     string_to_return = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
#
#     # strings --> offset_mapping
#     tokenized = tokenizer(
#             string_to_return,
#             add_special_tokens=False, 
#             return_offsets_mapping=True,
#             padding=True, 
#             return_tensors='pt'
#     ).to(token_ids.device)
#
#    return tokenized.offset_mapping

def batch_map_word_values(logits, batch_token_ids, strings, offset_mapping, is_pooled=False):

    if is_pooled:
        weights = logits.gather(1, batch_token_ids).cpu().numpy()
    else:
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

