import os
import sys
import torch
import numpy as np
import nltk
from transformers import AutoProcessor
from transformers import HfArgumentParser
from arguments import ModelArgs, DataArgs, TrainArgs
from datasets import load_dataset
from tools import random_mask, init_tokenizer

def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Config: processor
    processor = AutoProcessor.from_pretrained(
            model_args.processor_name,
            additional_special_tokens=[f"[unused{i}]" for i in range(100)]
    )
    processor = init_tokenizer(processor)

    # Config: modeling
    from models_mlsr_wgen import BlipForGenerativeEncoder
    model = BlipForGenerativeEncoder.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            lambda_d=0.00,
    )

    # Data: dataset
    dataset = load_dataset('json', data_files=data_args.train_file)['train']
    dataset = dataset.train_test_split(test_size=3000, seed=777)
    dataset = dataset.map(
            lambda x: {"title_masked": random_mask(x['title'], mask_p=data_args.title_mask_ratio)}
    )
    print(dataset)

    # Data: collator
    import datacollator 
    data_collator = datacollator.Product2Title(
            processor=processor,
            template_src=training_args.template_src,
            template_tgt=training_args.template_tgt,
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length,
            text_dropout=training_args.text_dropout,
            image_dropout=training_args.image_dropout,
            mask_decoder_inputs=True
    )

    from trainer import MyTrainer
    trainer = MyTrainer(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            processor=processor,
            kd=True,
            encoder_name_or_path="naver/splade-cocondenser-ensembledistil"
    )
    
    results = trainer.train()

    return results

if __name__ == '__main__':
    main()
