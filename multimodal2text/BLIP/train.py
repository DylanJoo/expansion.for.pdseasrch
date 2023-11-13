import os
import sys
from transformers import AutoProcessor, BlipForQuestionAnswering
from transformers import HfArgumentParser
from transformers import Trainer

from arguments import ModelArgs, DataArgs, TrainArgs
from datasets import load_dataset
import datacollator 

def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(model_args.processor_name)
    model = BlipForQuestionAnswering.from_pretrained(model_args.model_name_or_path)
    
    # Data: dataset
    dataset = load_dataset('json', data_files=data_args.train_file)['train']
    dataset = dataset.train_test_split(test_size=3000, seed=777)
    print(dataset)

    # Data: collator
    datacollator_classes = {"product2query": datacollator.Product2Query}
    data_collator = datacollator_classes[model_args.datacollator](
            processor=processor,
            image_dir=data_args.image_dir,
            template_src=training_args.template_src,
            template_tgt=training_args.template_tgt,
            max_src_length=data_args.max_src_length,
            max_tgt_length=data_args.max_tgt_length
    )

    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
    )
    
    results = trainer.train()

    return results

if __name__ == '__main__':
    main()
