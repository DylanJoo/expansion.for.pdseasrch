import torch
import numpy as np
import collections

def batch_transform_token_ids(tokenizer, batch_token_ids, return_attention_mask=False):
    tokens_to_add = []
    string_to_return = []

    # token_ids --> a token list and a sring
    for token_ids in batch_token_ids:
        decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_string = tokenizer.convert_tokens_to_string(decoded_tokens)
        tokens_to_add += decoded_tokens
        string_to_return.append(decoded_string)

    # strings --> offset_mapping
    tokenized = tokenizer(
            string_to_return,
            add_special_tokens=False, return_offsets_mapping=True,
            padding=True, return_tensors='pt'
    )

    ## [IMPORTANT] this is very ad-hoc way to deal with destriptive process of encoding
    if tokenized.input_ids.size(1) != batch_token_ids.size(1):
        tokenizer.add_tokens(list(set(tokens_to_add)))
        tokenized = tokenizer(
                string_to_return,
                add_special_tokens=False, return_offsets_mapping=True,
                padding=True, return_tensors='pt'
        )
        mapping_to_return = tokenized.offset_mapping
    else:
        mapping_to_return = tokenized.offset_mapping

    # check length
    if return_attention_mask:
        # mask = tokenized.attention_mask # derived from output token ids
        mask = (batch_token_ids != tokenizer.pad_token_id).long()
        return string_to_return, mapping_to_return, mask
    else:
        return string_to_return, mapping_to_return, None

def batch_map_word_values(logits, batch_token_ids, strings, offset_mapping, is_pooled=False):

    if is_pooled:
        logits = logits.unsqueeze(1).repeat((1, batch_token_ids.size(-1), 1))
    weights = logits.gather(2, batch_token_ids.unsqueeze(2)).squeeze(2).cpu().numpy()

    to_return = []
    for i, (offset, weight) in enumerate(zip(offset_mapping, weights)):
        word_id_to_weights = collections.defaultdict(float)
        words = strings[i]

        start, end = 0, 0
        for j, (start_, end_) in enumerate(offset):
            # retrieve the maximum between tokens
            if end_ != 0:
                try:
                    w = weight[j]
                except:
                    w = weight.mean()
                    print(f'use average {w} instead of unmatched weight')

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

