import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
from transformers import BlipForQuestionAnswering as BlipForQuestionAnswering_hf
from transformers.utils import ModelOutput

@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    losses: Optional[dict] = None
    loss: Optional[torch.FloatTensor] = None
    document_feat: Optional[torch.FloatTensor] = None
    product_feat: Optional[torch.FloatTensor] = None

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

class BlipForQuestionAnswering(BlipForQuestionAnswering_hf):

    def __init__(self, config, lambda_q=0, lambda_d=0.0001):
        super().__init__(config)
        self.pooling = splade_max
        self.pooling_q = splade_max
        self.lambda_q = lambda_q
        self.lambda_d = lambda_d
        self.ignored_token_ids = []

    def post_init(self):
        # self.decoder_start_token_id = 30522 # this has been initialized
        # this is as same as tokenizer.bos_token_id

        self.text_encoder_start_token_id = 30523
        # this is as same as tokenizer.enc_token_id

        self.text_encoder_cls_token_id = 101
        # this is as same as tokenizer.enc_token_id

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

        ## Image encoding
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        ## Text product encoding
        product_embeds_0 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        ## Multimodal image-text encoding
        product_embeds_1 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )[0]

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            labels_inputs = labels.masked_fill(labels == -100, self.text_decoder.config.pad_token_id)
            decoder_input_ids = labels_inputs.clone()

        decoder_input_ids[:, 0] = self.decoder_start_token_id

        ### [DEC] as decoding start token
        #### decode from text-only encoding
        product_outputs_0 = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=product_embeds_0,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )
        product_logits_0 = product_outputs_0.logits
        product_result_0 = self.pooling(
                product_logits_0, decoder_attention_mask # since query has mask by labels, here is fine for this.
        )[:, :-2] 

        #### decode from image-text encoding
        product_outputs_1 = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=product_embeds_1,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )
        product_logits_1 = product_outputs_1.logits
        product_result_1 = self.pooling(
                product_logits_1, decoder_attention_mask 
        )[:, :-2]

        # loss of generation
        if (product_outputs_0 is not None) and (product_outputs_1 is not None):
            loss_mtlm = (product_outputs_0.loss + product_outputs_1.loss) / 2 * 0
        else:
            loss_mtlm = 0

        # loss for regularization
        L1Loss = FLOPS()
        loss_reg = (L1Loss(product_result_1) + L1Loss(product_result_0)) * self.lambda_d
        anti_zero = 1/(torch.sum(product_result_0)**2) + 1/(torch.sum(product_result_1)**2)

        return BlipTextVisionModelOutput(
                loss = loss_mtlm + loss_reg,
                document_feat=product_result_0,
                product_feat=product_result_1
        )
