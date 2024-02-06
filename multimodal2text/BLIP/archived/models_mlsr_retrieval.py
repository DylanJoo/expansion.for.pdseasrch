import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from torch.nn.functional import normalize
import torch.nn as nn

from transformers.utils import ModelOutput
from transformers import BlipForImageTextRetrieval 
from transformers.models.blip.modeling_blip_text import BlipTextOnlyMLMHead

@dataclass
class BlipForRetrievalOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    image_feat: Tuple[torch.FloatTensor] = None
    document_feat: Tuple[torch.FloatTensor] = None
    product_feat: Tuple[torch.FloatTensor] = None

def splade_max(logits, attention_mask=None, labels_=None):
    """
    param logits: the logit values indicate the token distribution before log().
    param attention_mask: the mask for dimension of token length, the shape will be (B, L, 1)
    param label_mask: the mask for dimension of vocabulary, the shape will be (B, 1, V)
    """
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

class BlipForRetrieval(BlipForImageTextRetrieval):

    def __init__(self, config):
        super().__init__(config)
        self.cls = BlipTextOnlyMLMHead(config.text_config) 
        self.cls.to(self.device)
        self.pooling = splade_max

    def load_pretrained_cls_embeddings(self, model_name_or_path):
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        q_input_ids: Optional[torch.LongTensor] = None,
        q_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipForRetrievalOutputs]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # text-only representation
        text_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )[0]

        # text-image representation
        multimodal_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )[0]

        # for dense retrieval (get cls tokens)
        # image_dense_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # text_dense_feat = normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        # multimodal_dense_feat = normalize(self.text_proj(multimodal_embeds[:, 0, :]), dim=-1)
        # loss_dense = image_dense_feat @ text_dense_feat.t()

        # for sparse retrieval (get max token logits)
        image_sparse_feat = self.pooling(self.cls(image_embeds))[:, :-2]
        text_sparse_feat = self.pooling(self.cls(text_embeds), attention_mask)[:, :-2]
        multimodal_sparse_feat = self.pooling(self.cls(multimodal_embeds), attention_mask)[:, :-2]

        return BlipForRetrievalOutputs(
            loss=0,
            document_feat=text_sparse_feat, 
            image_feat=image_sparse_feat,
            product_feat=multimodal_sparse_feat,
        )
