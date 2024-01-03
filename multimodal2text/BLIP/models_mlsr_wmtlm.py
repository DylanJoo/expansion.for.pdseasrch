import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
from transformers import BlipForQuestionAnswering as BlipForQuestionAnswering_hf
from transformers.utils import ModelOutput

@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    losses: Optional[dict] = None

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class BlipForQuestionAnswering(BlipForQuestionAnswering_hf):

    def post_init(self):
        # self.decoder_start_token_id = 30522 # this has been initialized
        # this is as same as tokenizer.bos_token_id

        self.text_encoder_start_token_id = 30523
        # this is as same as tokenizer.enc_token_id

        self.text_encoder_cls_token_id = 101
        # this is as same as tokenizer.enc_token_id

        # regularization
        self.lambda_q = 1e-1
        self.lambda_d = 1e-2

    @staticmethod
    def splade_max(logits, attention_mask=None):
        # tokens: output of a huggingface tokenizer
        relu = nn.ReLU(inplace=False)
        if attention_mask is not None:
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

        # product text encoding
        product_embeds_0 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        # product image-text encoding
        product_embeds_1 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )[0]

        # query text encoding
        labels_for_encoder = labels.clone()
        labels_for_encoder[:, 0] = self.text_encoder_start_token_id
        query_embeds = self.text_encoder(
            input_ids=labels_for_encoder,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=return_dict,
        )[0]

        ## [CLS] as start token
        cls_ids = torch.full(
            (input_ids.size(0), 1), fill_value=self.text_encoder_cls_token_id, device=input_ids.device
        )

        # product decoding from product text encoding
        product_logits_0 = self.text_decoder(
            input_ids=cls_ids,
            attention_mask=None,
            encoder_hidden_states=product_embeds_0,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        # product decoding from product image-text encoding
        product_logits_1 = self.text_decoder(
            input_ids=cls_ids,
            attention_mask=None,
            encoder_hidden_states=product_embeds_1,
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        # query decoding from query text encoding
        query_logits = self.text_decoder(
            input_ids=cls_ids,
            attention_mask=None,
            encoder_hidden_states=query_embeds,
            encoder_attention_mask=decoder_attention_mask,
            return_dict=return_dict,
            reduction="none",
        ).logits

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        ## [DEC] as decoding start token
        # query generation from text representation
        product_outputs_0_kd = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=product_embeds_0,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        # query generation from text-iamge representation
        product_outputs_1_kd = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=product_embeds_1,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        loss_mtlm = product_outputs_0_kd.loss + product_outputs_1_kd.loss
        product_logits_0_kd = product_outputs_0_kd.logits
        product_logits_1_kd = product_outputs_1_kd.logits

        # reshape 
        query_result = self.splade_max(query_logits[:, 0, :]) ##(bsz, Vocab)
        product_result_0 = self.splade_max(product_logits_0[:, 0, :]) ##(bsz, Vocab)
        product_result_1 = self.splade_max(product_logits_1[:, 0, :]) ##(bsz, Vocab)
        product_result_0_kd = self.splade_max(product_logits_0_kd[:, :-1, :], decoder_attention_mask[:, :-1]) ##(bsz, Vocab)
        product_result_1_kd = self.splade_max(product_logits_1_kd[:, :-1, :], decoder_attention_mask[:, :-1]) ##(bsz, Vocab)

        # loss of splade
        CELoss = nn.CrossEntropyLoss()
        scores_0 = torch.mm(query_result, torch.permute(product_result_0,[1,0])) # shape (bsz, bsz)
        scores_1 = torch.mm(query_result, torch.permute(product_result_1,[1,0])) # shape (bsz, bsz)
        scores_pair = torch.cat([scores_0.diag().unsqueeze(1), scores_1.diag().unsqueeze(1)], dim=-1) # shape (bsz, 2)
        loss_0 = CELoss(scores_0, torch.arange(0, scores_0.size(0), device=scores_0.device))
        loss_1 = CELoss(scores_1, torch.arange(0, scores_1.size(0), device=scores_1.device)) 
        loss_pair = CELoss(scores_pair, torch.ones(scores_pair.size(0), dtype=torch.long, device=scores_pair.device)) 

        # loss of knowledge distillation
        MSELoss = nn.MSELoss()
        loss_0_kd = MSELoss(product_result_0, product_result_0_kd)
        loss_1_kd = MSELoss(product_result_1, product_result_1_kd)

        # loss for regularization
        L1Loss = FLOPS()
        loss_reg_q = L1Loss(query_result) * self.lambda_q
        loss_reg_d = (L1Loss(product_result_1) + L1Loss(product_result_0)) * self.lambda_d

        return BlipTextVisionModelOutput(
            loss=loss_0+loss_1+loss_pair+loss_mtlm+loss_1_kd+loss_0_kd+loss_reg_q+loss_reg_d,
            losses={
                'splade_loss_0': loss_0, 
                'splade_loss_1': loss_1, 
                'splade_loss_pair': loss_pair,
                'generation_0': product_outputs_0_kd.loss, 
                'generation_1': product_outputs_1_kd.loss,
                'self_kd_loss_0': loss_0_kd,
                'self_kd_loss_1': loss_1_kd,
                'reg_loss_q': loss_reg_q,
                'reg_loss_d': loss_reg_d/2,
            }
        )
