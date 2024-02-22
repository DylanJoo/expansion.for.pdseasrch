import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Trainer
from transformers import AutoModelForMaskedLM
from utils import splade_max

def mask_entities(logits, labels_):
    mask = torch.ones(logits.size(0), logits.size(-1)).to(logits.device)
    mask.scatter_(-1, labels_, 0)
    logits = logits * mask
    return logits

def show_entities(logits, labels_):
    show = torch.zeros(logits.size(0), logits.size(-1)).to(logits.device)
    show.scatter_(-1, labels_, 1)
    logits = logits * show
    return logits

class MyTrainer(Trainer):
    def __init__(self, processor=None, kd=False, encoder_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.kd = kd
        self.pooling = splade_max

        if kd:
            self.encoder = AutoModelForMaskedLM.from_pretrained(encoder_name_or_path).to(self.args.device)
            self.encoder.eval()
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_gen, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = loss_gen
        labels_clean = inputs['labels'].clone()
        labels_clean = labels_clean.masked_fill(labels_clean == -100, self.processor.tokenizer.pad_token_id)

        # compute loss for KD
        if self.kd:
            query_feat = splade_max(
                    self.encoder(input_ids=labels_clean, attention_mask=inputs['decoder_attention_mask']).logits, 
                    inputs['decoder_attention_mask']
            )

            CELoss = nn.CrossEntropyLoss()
            # contrastive learning for document encoder
            scores_0 = query_feat @ torch.permute(outputs.document_feat,[1,0])
            scores_1 = query_feat @ torch.permute(outputs.product_feat,[1,0])
            loss_0 = CELoss(scores_0, torch.arange(0, scores_0.size(0), device=scores_0.device))
            loss_1 = CELoss(scores_1, torch.arange(0, scores_1.size(0), device=scores_1.device))
            loss += loss_0 + loss_1

            # distill entity-level tokens
            query_ent_feat = show_entities(query_feat, labels_clean)
            scores_ent_0 = query_ent_feat @ torch.permute(outputs.document_feat,[1,0])
            scores_ent_1 = query_ent_feat @ torch.permute(outputs.product_feat,[1,0])
            loss_ent_0 = CELoss(scores_ent_0, torch.arange(0, scores_ent_0.size(0), device=scores_ent_0.device))
            loss_ent_1 = CELoss(scores_ent_1, torch.arange(0, scores_ent_1.size(0), device=scores_ent_1.device))
            loss += loss_ent_0 + loss_ent_1

            # distill remainig tokens
            query_rest_feat = mask_entities(query_feat, labels_clean)
            scores_rest_0 = query_rest_feat @ torch.permute(outputs.document_feat,[1,0])
            scores_rest_1 = query_rest_feat @ torch.permute(outputs.product_feat,[1,0])
            scores_pair = torch.cat([scores_rest_0.diag().unsqueeze(1), scores_rest_1.diag().unsqueeze(1)], dim=-1) # shape (bsz, 2)
            loss_pair = CELoss(scores_pair, torch.ones(scores_pair.size(0), dtype=torch.long, device=scores_pair.device))
            loss += loss_pair

        # check losses here
        if self.state.global_step % 10 == 0:

            # print losses
            print('generation_loss_(--> labels)', loss_gen.item())
            print('lsr_token_loss_(q*, prod-t)', loss_0.item())
            print('lsr_token_loss_(q*, prot-it)', loss_1.item())
            print('lsr_word_loss_(q*, prod-t)', loss_ent_0.item())
            print('lsr_word_loss_(q*, prot-it)', loss_ent_1.item())
            print('visual_enhanced_loss', loss_pair.item())

            # query words
            tokens = self.processor.tokenizer.batch_decode(
                    labels_clean[:5], skip_special_tokens=True
            )
            print('\nquery (words): '.join([""] + tokens) + '\n')

            # query MLM
            query_rest_feat = mask_entities(query_feat, labels_clean)
            top_k = torch.topk(query_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nquery (tokens): '.join([""] + top_k_tokens) + '\n')

            # product MLM
            top_k = torch.topk(outputs.product_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :],
                    skip_special_tokens=True
            )
            print('\ndoc (tokens): '.join([""] + top_k_tokens) + '\n')

            # query prediction
            token_ids = torch.max(outputs.product_logit[:5], dim=-1).indices
            tokens = self.processor.tokenizer.batch_decode(
                    token_ids.detach().cpu().numpy()[:, :], 
                    skip_special_tokens=True
            )
            print('\ndoc (words): '.join([""] + tokens) + '\n')

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
