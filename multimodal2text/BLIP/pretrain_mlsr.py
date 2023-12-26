import os
import sys
import torch
import numpy as np
import nltk
from transformers import AutoProcessor
from transformers import HfArgumentParser
from transformers import Trainer
from arguments import ModelArgs, DataArgs, TrainArgs
from datasets import load_dataset
from tools import random_mask
import datacollator 

def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # from models import BlipForQuestionAnswering
    from models_mlsr_prt import BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(model_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_args.processor_name)
    
    # Data: dataset
    dataset = load_dataset('json', data_files=data_args.train_file)['train']
    dataset = dataset.train_test_split(test_size=3000, seed=777)
    # Data: augmentation: title word drop
    dataset = dataset.map(
            lambda x: {"title_masked": random_mask(x['title'], mask_p=data_args.title_mask_ratio)}
    )  

    print(dataset)
    print(dataset['train'][0])

    # Data: collator
    data_collator = datacollator.Product2Title(
            processor=processor,
            template_src=training_args.template_src,
            template_tgt=training_args.template_tgt,
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length
    )

    # Train: 
    ## eval metrics
    import evaluate
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    @torch.no_grad()
    def compute_metrics(eval_preds):
        """
        In VQA, the model return is vision hidden embeddings
        """
        output, labels = eval_preds
        output = output[0]
        preds = np.argmax(output, axis=-1)
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
        decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != processor.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    
    results = trainer.train()

    return results

if __name__ == '__main__':
    main()
