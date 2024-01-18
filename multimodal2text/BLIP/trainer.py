import torch
from transformers import Trainer

class MyTrainer(Trainer):
    def __init__(self, processor=None, **kwargs):
        self.processor = processor
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        to_return = super().compute_loss(model, inputs, return_outputs=True)

        # check losses here
        if self.state.global_step % 50 == 0:
            for k, v in to_return[1].losses.items():
                print(k, v.item())

            tokens = self.processor.tokenizer.batch_decode(
                    inputs['labels'][:5], skip_special_tokens=True
            )
            print('\nl: ' + '\nl: '.join(tokens) + '\n')

            top_k = torch.topk(to_return[1].query_logits[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nq: ' + '\nq: '.join(top_k_tokens) + '\n')

            top_k = torch.topk(to_return[1].document_logits[:5], k=5, dim=-1).indices
            top_k_tokens = self.processor.tokenizer.batch_decode(
                    top_k.detach().cpu().numpy()[:, :]
            )
            print('\nd: ' + '\nd: '.join(top_k_tokens) + '\n')

        if return_outputs:
            return to_return
        else:
            return to_return[0]
