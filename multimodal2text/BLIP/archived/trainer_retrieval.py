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

class MyTrainer(Trainer):
    def __init__(self, processor=None, kd=False, encoder_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.kd = kd
        self.encoder = AutoModelForMaskedLM.from_pretrained(
                encoder_name_or_path
        ).to(self.args.device)
        self.encoder.eval()
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # compute loss for KD
        if self.kd:
            q_teacher_feat = self.model.pooling(self.encoder(
                input_ids=inputs['q_input_ids'], 
                attention_mask=inputs['q_attention_mask']
            ).logits)
            d_teacher_feat = self.model.pooling(self.encoder(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            ).logits)
            scores_teacher = q_teacher_feat @ d_teacher_feat.t()

            scores = q_teacher_feat @ outputs.product_feat.t()

            MSELoss = nn.MSELoss()
            kd_loss = MSELoss(scores_teacher, scores)
            loss += kd_loss

        # check losses here
        if self.state.global_step % 10 == 0:
            print('loss kd: ', kd_loss)

            # query MLM & product MLM
            top_k = torch.topk(q_teacher_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nq*: ' + '\nq*: '.join(top_k_tokens) + '\n')

            top_k = torch.topk(outputs.product_feat[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nd: ' + '\nd: '.join(top_k_tokens) + '\n')

        if return_outputs:
            return (loss, outputs)
        else:
            return loss
