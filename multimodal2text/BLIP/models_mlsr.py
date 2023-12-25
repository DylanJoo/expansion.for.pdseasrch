import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
from transformers import BlipForQuestionAnswering as BlipForQuestionAnswering_hf
from transformers.utils import ModelOutput

@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    losses: Tuple[torch.FloatTensor] = None

class BlipForQuestionAnswering(BlipForQuestionAnswering_hf):

    @staticmethod
    def splade_max(logits, attention_mask=None):
        # tokens: output of a huggingface tokenizer
        relu = nn.ReLU(inplace=False)
        if attention_mask:
            values, _ = torch.max(torch.log(1 + relu(logits)) * attention_mask.unsqueeze(-1), dim=1)
        else:
            values = torch.log(1 + relu(logits))
        return values    

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

        # text representation
        product_embeds_0 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        # text representation (query)
        query_embeds = self.text_encoder(
            input_ids=labels,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        # image-text representation
        product_embeds_1 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )[0]

        # [original] this is the right shifted labels
        bos_ids = torch.full(
            (input_ids.size(0), 1), fill_value=self.decoder_start_token_id, device=input_ids.device
        )

        # decoding from text representation
        product_logits_0 = self.text_decoder(
            input_ids=bos_ids,
            attention_mask=None,
            encoder_hidden_states=product_embeds_0,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        # decoding from text representation
        query_logits = self.text_decoder(
            input_ids=bos_ids,
            attention_mask=None,
            encoder_hidden_states=query_embeds,
            encoder_attention_mask=decoder_attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        # decoding from image-text representation
        product_logits_1 = self.text_decoder(
            input_ids=bos_ids,
            attention_mask=None,
            encoder_hidden_states=product_embeds_1,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        # reshape
        query_result = self.splade_max(query_logits.view(-1,query_logits.size(-1))) ##(bsz, Vocab)
        product_result_0 = self.splade_max(product_logits_0.view(-1,product_logits_0.size(-1))) ##(bsz, Vocab)
        product_result_1 = self.splade_max(product_logits_1.view(-1,product_logits_1.size(-1))) ##(bsz, Vocab)

        # scoring
        scores_0 = torch.mm(query_result, torch.permute(product_result_0,[1,0])) # shape (bsz, bsz)
        scores_1 = torch.mm(query_result, torch.permute(product_result_1,[1,0])) # shape (bsz, bsz)
        scores_pair = torch.cat([scores_0.diag().unsqueeze(1), scores_1.diag().unsqueeze(1)], dim=-1) # shape (bsz, 2)

        # losses
        CELoss = nn.CrossEntropyLoss()
        loss_0 = CELoss(scores_0, torch.arange(0, scores_0.size(0), device=scores_0.device))
        loss_1 = CELoss(scores_1, torch.arange(0, scores_1.size(0), device=scores_1.device)) 
        loss_pair = CELoss(scores_pair, torch.ones(scores_pair.size(0), dtype=torch.long, device=scores_pair.device)) 

        return BlipTextVisionModelOutput(
            loss=loss_0+loss_1+loss_pair,
            losses=(loss_0, loss_1, loss_pair)
        )
