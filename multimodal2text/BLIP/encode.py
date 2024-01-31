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
from pyserini.encode import SpladeQueryEncoder

@dataclass
class BlipBiEncoder(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    embeds: Optional[torch.FloatTensor] = None

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
        self.pooling = pooling

        # Initialize weights and apply final processing
        self.post_init()

    def encode(self, titles: List[str], descriptions: List[str], images_path: List[str] = None, **processor_kwargs):
        texts = [f"{t} [SEP] {d}" for t, d in zip(titles, descriptions)]
        images = []
        for img in images_path:
            try:
                images.append(
                        Image.open(img).convert('RGB').resize((384, 384))
                )
            except:
                blank = Image.new('RGB', (384, 384), color=(255, 255, 255))
                images.append(blank)

        inputs = self.processor(
                images=images, 
                text=texts,
                return_tensors='pt',
                return_attention_mask=True,
                truncation=True,
                padding=True,
                **processor_kwargs
        ).to(self.device)
        inputs['input_ids'][:, 0] = self.processor.tokenizer.enc_token_id

        values = self.lsr_max(self.forward(**inputs).logits)
        # values = values.cpu().detach().numpy() 
        # not return numpy because when generate logits, the torch.nonzero is better than numpy's
        values = values.cpu().detach()
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
        ).logits[:, 0, :-2]

        return BlipBiEncoder(
                logits=product_logits_1,
                embeds=product_embeds_1
        )

