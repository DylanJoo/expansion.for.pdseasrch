from transformers import Trainer as Trainer_hf

class Trainer(Trainer_hf):

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.state.global_step and self.state.global_step % 50 == 0:
            self.verbose(model, **inputs)

        return (loss, outputs) if return_outputs else loss
    
    def verbose(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            outputs = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
            )
            
            decoded_texts = model.tokenizer

