from transformers import Trainer

class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        to_return = super().compute_loss(model, inputs, return_outputs=True)

        # check losses here
        if self.state.global_step % 50 == 0:
            for k, v in to_return[1].losses.items():
                print(k, v.detach().cpu().numpy().round(3))

        if return_outputs:
            return to_return
        else:
            return to_return[0]
