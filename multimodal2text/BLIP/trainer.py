import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Trainer
from transformers import AutoModelForMaskedLM

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

def mask_entities(logits, labels_):
    mask = torch.ones(logits.size(0), logits.size(-1)).to(logits.device)
    mask.scatter_(-1, labels_, 0)
    logits = logits * mask
    return logits

class MyTrainer(Trainer):
    def __init__(self, processor=None, kd=False, encoder_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.kd = kd
        self.encoder = AutoModelForMaskedLM.from_pretrained(encoder_name_or_path).to(self.args.device)
        self.encoder.eval()
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_gen, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = loss_gen

        # compute loss for KD
        if self.kd:
            labels_inputs = inputs['labels'].masked_fill(inputs['labels'] == -100, self.processor.tokenizer.pad_token_id)
            q_star_feat = self.model.pooling(
                    self.encoder(input_ids=labels_inputs, attention_mask=inputs['decoder_attention_mask']).logits
            )

            scores_0 = q_star_feat @ torch.permute(outputs.document_feat,[1,0])
            scores_1 = q_star_feat @ torch.permute(outputs.product_feat,[1,0])

            CELoss = nn.CrossEntropyLoss()
            loss_0 = CELoss(scores_0, torch.arange(0, scores_0.size(0), device=scores_0.device))
            loss_1 = CELoss(scores_1, torch.arange(0, scores_1.size(0), device=scores_1.device))
            loss += loss_0 + loss_1

            # distill entity-level tokens
            scores_rest_0 = q_star_feat @ torch.permute(
                    mask_entities(outputs.document_feat, labels_inputs),[1,0]
            )
            scores_rest_1 = q_star_feat @ torch.permute(
                    mask_entities(outputs.product_feat, labels_inputs),[1,0]
            )
            scores_rest_pair = torch.cat([scores_rest_0.diag().unsqueeze(1), scores_rest_1.diag().unsqueeze(1)], dim=-1) # shape (bsz, 2)
            loss_pair = CELoss(scores_rest_pair, torch.ones(scores_rest_pair.size(0), dtype=torch.long, device=scores_rest_pair.device))
            loss += loss_pair

        # check losses here
        if self.state.global_step % 10 == 0:

            if 'labels' in inputs:
                labels_clean = inputs['labels'].clone()
                labels_clean = labels_clean.masked_fill(labels_clean == -100, self.processor.tokenizer.pad_token_id)
                tokens = self.processor.tokenizer.batch_decode(
                        labels_clean[:5], skip_special_tokens=True
                )
                print('\nl: ' + '\nl: '.join(tokens) + '\n')

            print('generation_loss_(--> labels)', loss_gen.item())
            print('splade_loss_(q*, prod-t)', loss_0.item())
            print('splade_loss_(q*, prot-it)', loss_1.item())
            print('visual_enhanced_loss', loss_pair.item())

            # query MLM
            top_k = torch.topk(q_star_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nq*: ' + '\nq*: '.join(top_k_tokens) + '\n')

            # product MLM
            top_k = torch.topk(outputs.product_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nd: ' + '\nd: '.join(top_k_tokens) + '\n')

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
