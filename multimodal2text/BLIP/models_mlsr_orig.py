# [NOTE] Unfinished
import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers import BlipForQuestionAnswering as BlipForQuestionAnswering_hf
from transformers.utils import ModelOutput
from transformers import BlipForImageTextRetrieval 
from transformers.models.blip.modeling_blip_text import BlipTextOnlyMLMHead

@dataclass
class BlipForRetrievalOutouts(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    image_features: Tuple[torch.FloatTensor] = None
    text_features: Tuple[torch.FloatTensor] = None
    multimodal_features: Tuple[torch.FloatTensor] = None

class BlipForRetrieval(BlipForImageTextRetrieval):

    def post_init(self, model_name_or_path):
        # add a masked language model head here.
        self.cls = BlipTextOnlyMLMHead.from_pretrained(model_name_or_path)
        self.cls.to(self.device)
        self.cls.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:

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
        image_dense_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_dense_feat = normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        loss_dense = image_dense_feat @ text_dense_feat.t()

        # for sparse retrieval (get max token logits)
        image_sparse_feat = self.cls(image_embeds)
        text_sparse_feat = self.cls(text_embeds)
        multimodal_sparse_feat = self.cls(multimodal_embeds)

        return BlipForRetrievalOutouts(
            loss=decoder_loss,
            lm_logits=lm_logits,
            image_features=(image_dense_feat, image_sparse_feat),
            text_features=(text_dense_feat, text_sparse_feat),
            multimodal_features=(multimodal_dense_feat, multimodal_sparse_feat),
        )
