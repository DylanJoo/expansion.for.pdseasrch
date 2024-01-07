import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List
import numpy as np
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers.models.blip.modeling_blip import BlipTextLMHeadModel, BlipTextModel, BlipVisionModel
from transformers import BlipConfig, BlipPreTrainedModel, AutoProcessor
from PIL import Image
from tools import init_tokenizer

@dataclass
class BlipBiEncoder(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    embeds: Optional[torch.FloatTensor] = None

class BlipForQueryEncoder(BlipPreTrainedModel):

    @staticmethod
    def lsr_max(logits):
        relu = nn.ReLU(inplace=False)
        values = torch.log(1 + relu(logits))
        return values    

    def post_init(self):
        # self.decoder_start_token_id = 30522 # this has been initialized
        # this is as same as tokenizer.bos_token_id

        self.text_encoder_start_token_id = 30523
        # this is as same as tokenizer.enc_token_id

        self.text_encoder_cls_token_id = 101
        # this is as same as tokenizer.enc_token_id

        # clone from pyserini splade query encoder
        self.reverse_voc = {v: k for k, v in self.processor.tokenizer.vocab.items()}
        self.weight_range = 5
        self.quant_range = 256

    def __init__(self, config: BlipConfig, processor_name=None, pooling='max'):
        super().__init__(config)

        processor = AutoProcessor.from_pretrained(processor_name or config.name_or_path)
        self.processor = init_tokenizer(processor)
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)
        self.text_decoder = BlipTextLMHeadModel(config.text_config)
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id
        self.pooling = pooling

        # Initialize weights and apply final processing
        self.post_init()

    def encode(self, queries: List[str], **processor_kwargs):
        texts = [f"{q}" for q in queries]
        inputs = self.processor(
                text=texts, 
                truncation=True,
                padding=True,
                return_tensors='pt', 
                **processor_kwargs
        ).to(self.device)

        # replace the first token with [encode]
        inputs['input_ids'][:, 0] = self.processor.tokenizer.enc_token_id
        values = self.lsr_max(self.forward(**inputs).logits)
        values = values.cpu().detach().numpy()
        raw_weights = self._output_to_weight_dicts(values)
        return self._get_encoded_query_token_weight_dicts(raw_weights)[0]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # text representation (query)
        query_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )[0]

        cls_ids = torch.full(
            (input_ids.size(0), 1), fill_value=self.text_encoder_cls_token_id, device=input_ids.device
        )

        # decoding from text representation
        query_logits = self.text_decoder(
            input_ids=cls_ids,
            attention_mask=None,
            encoder_hidden_states=query_embeds,
            encoder_attention_mask=attention_mask,
            reduction="none",
        ).logits

        return BlipBiEncoder(
                logits=query_logits[:, 0, :], # aggregated at first token
                embeds=query_embeds
        )
    def _output_to_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
            to_return.append(d)
        return to_return

    def _get_encoded_query_token_weight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / self.weight_range * self.quant_range)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return

class BlipForProductEncoder(BlipPreTrainedModel):

    @staticmethod
    def lsr_max(logits):
        relu = nn.ReLU(inplace=False)
        values = torch.log(1 + relu(logits))
        return values    

    def post_init(self):
        # self.decoder_start_token_id = 30522 # this has been initialized
        # this is as same as tokenizer.bos_token_id

        self.text_encoder_start_token_id = 30523
        # this is as same as tokenizer.enc_token_id

        self.text_encoder_cls_token_id = 101
        # this is as same as tokenizer.enc_token_id

    def __init__(self, config: BlipConfig, processor_name=None, pooling='max'):
        super().__init__(config)

        processor = AutoProcessor.from_pretrained(processor_name or config.name_or_path)
        self.processor = init_tokenizer(processor)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)
        self.text_decoder = BlipTextLMHeadModel(config.text_config)
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id
        self.pooling = pooling

        # Initialize weights and apply final processing
        self.post_init()

    def encode(self, titles: List[str], descriptions: List[str], images_path: List[str] = None, **processor_kwargs):
        texts = [f"title: {t} context: {d}" for t, d in zip(titles, descriptions)]
        images = []
        for img in images_path:
            try:
                images.append(Image.open(img.convert('RGB').resize((384, 384))))
            except:
                blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
                images.append(blank)

        inputs = self.processor(
                images=images, text=texts,
                truncation=True,
                padding=True,
                return_tensors='pt',
                **processor_kwargs
        ).to(self.device)

        # replace the first with [ENC]
        inputs['input_ids'][:, 0] = self.processor.tokenizer.enc_token_id
        values = self.lsr_max(self.forward(**inputs).logits)
        values = values.cpu().detach().numpy()
        return values

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # image representation
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # image-text representation
        product_embeds_1 = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]

        cls_ids = torch.full(
            (input_ids.size(0), 1), fill_value=self.text_encoder_cls_token_id, device=input_ids.device
        )

        # decoding from image-text representation
        product_logits_1 = self.text_decoder(
            input_ids=cls_ids,
            attention_mask=None,
            encoder_hidden_states=product_embeds_1,
            encoder_attention_mask=attention_mask,
            reduction="none",
        ).logits

        return BlipBiEncoder(
                logits=product_logits_1[:, 0, :], # aggregated at first token
                embeds=product_embeds_1
        )

