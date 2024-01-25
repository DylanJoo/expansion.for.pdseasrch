import copy
import torch
import numpy as np
import collections
import torch.nn.functional as F

def batch_transform_token_ids(tokenizer, batch_token_ids):
    string_to_return = []

    # token_ids --> a token list and a sring
    string_to_return = tokenizer.batch_decode(
            batch_token_ids,
            skip_special_tokens=True
    )

    # strings --> offset_mapping
    tokenized = tokenizer(
            string_to_return,
            add_special_tokens=False, return_offsets_mapping=True,
            padding=True, return_tensors='pt', 
    ).to(batch_token_ids.device)

    ## [IMPORTANT] this is very ad-hoc way to deal with destriptive process of encoding
    ### We can use the new tokenized outputs tokens as word features
    # if tokenized.input_ids.size(1) != batch_token_ids.size(1):
    #     tokenizer_ = copy.deepcopy(tokenizer)
    #     # get the length offset, and make two matrices same size
    #     offset = batch_token_ids.size(1) - tokenized.input_ids.size(1) # e.g., 5 - 7
    #     truncated = F.pad(tokenized.input_ids, (0, offset), mode='replicate').to(batch_token_ids.device)
    #     # get the different token ids
    #     offset_tokens = batch_token_ids[(batch_token_ids != truncated)].flatten()
    #     tokens_to_add = tokenizer.convert_ids_to_tokens(offset_tokens)
    #     tokenizer_.add_tokens(tokens_to_add)
    #     tokenized = tokenizer_(
    #             string_to_return,
    #             add_special_tokens=False, return_offsets_mapping=True,
    #             padding=True, return_tensors='pt'
    #     )
    #     mapping_to_return = tokenized.offset_mapping
    # else:
    #     mapping_to_return = tokenized.offset_mapping

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

