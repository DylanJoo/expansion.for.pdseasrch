import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
from torch.nn import CrossEntropyLoss
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
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
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

        # text-only representation
        question_embeds_0 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        # text-image representation
        question_embeds_1 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )[0]

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        # decoding from text-only representation
        batch_size = decoder_input_ids.shape[0]
        answer_output_0 = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds_0,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )
        lm_logits_0 = answer_output_0.logits

        # decoding from text-image representation
        answer_output_1 = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds_1,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )
        lm_logits_1 = answer_output_1.logits

        lm_logits = lm_logits_1 - lm_logits_0

        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous().to(shifted_prediction_scores.device)
            loss_fct = CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
            decoder_loss = loss_fct(shifted_prediction_scores.view(-1, self.text_decoder.config.vocab_size), labels.view(-1))
            decoder_loss += (answer_output_0.loss.mean() + answer_output_1.loss.mean())
        else:
            decoder_loss = None

        if not return_dict:
            outputs = (decoder_loss, lm_logits, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return BlipTextVisionModelOutput(
            loss=decoder_loss,
            lm_logits=lm_logits,
            image_embeds=image_embeds,
            last_hidden_state=None,
            hidden_states=None,
            attentions=None
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:


        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)

        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        question_embeds = question_outputs[0]

        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)

        bos_ids = torch.full(
            (question_embeds.size(0), 1), fill_value=self.decoder_start_token_id, device=question_embeds.device
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs
