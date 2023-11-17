import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers import BlipForQuestionAnswering as BlipForQuestionAnswering_hf
from transformers.utils import ModelOutput

@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BlipForQuestionAnswering(BlipForQuestionAnswering_hf):

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:

        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either `decoder_input_ids` or `labels` should be passed when calling `forward` with"
                " `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you"
                " are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`"
            )

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

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )
        lm_logits = answer_output.logits

        if labels is not None:
            decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
        else:
            decoder_loss = None

        if not return_dict:
            outputs = (decoder_loss, lm_logits, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return BlipTextVisionModelOutput(
            loss=decoder_loss,
            lm_logits=lm_logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )
