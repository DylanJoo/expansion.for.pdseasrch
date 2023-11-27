import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import TrainingArguments, Trainer, SchedulerType
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import Dataset
import torch
from PIL import Image
from evaluate import load
import pandas as pd
import numpy as np

random_seed = 42
np.random.seed(random_seed)

# path for the images
IMG_FILE_PATH = "/home/jhju/datasets/pdsearch/images/"

# checkpoint for pretrained model
CHECKPOINT = "microsoft/git-large-coco"

PROCESSOR = AutoProcessor.from_pretrained(CHECKPOINT)
MODEL = AutoModelForCausalLM.from_pretrained(CHECKPOINT)

# Defining the transformation function
def transforms(example_batch):
    images = [Image.open(x).convert("RGB") for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = PROCESSOR(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

def filter_missing_images(df: pd.DataFrame):
    global IMG_FILE_PATH
    # Convert the product column to string if it's not already
    df["product"] = df["product"].astype(str)
    # Create a new column with the full image path for ease of access
    df['image_path'] = IMG_FILE_PATH + df["product"] + '.jpg'

    # Check which image paths do not exist
    mask = df['image_path'].apply(os.path.exists)

    # Work with a copy to avoid SettingWithCopyWarning
    df_cleaned = df.loc[mask].copy()

    # Drop the auxiliary column 'image_path'
    df_cleaned.drop(columns=['image_path'], inplace=False, errors='ignore')

    return df_cleaned

def prepare_data():
    qrels_df = pd.read_csv("/home/jhju/datasets/pdsearch/product-search-train-filtered.qrels", sep='\t', header=None)
    qrels_df.columns = ['qid', 'nothing', 'product', 'relevance']
    relevant_pair_df = qrels_df[qrels_df['relevance'] >= 2]
    relevant_pair_df.drop(columns=['nothing', 'relevance'], inplace=True)
    
    # Reset the index of the filtered DataFrame if needed
    relevant_pair_df = filter_missing_images(relevant_pair_df)
    relevant_pair_df.reset_index(drop=True, inplace=True)

    # Randomly select 3000 rows as validation data
    total_rows = len(relevant_pair_df)
    
    # Define the number of rows you want for validation
    num_validation_rows = 3000
    
    # Generate random indices to select rows for validation
    validation_indices = np.random.choice(total_rows, num_validation_rows, replace=False)

    # Create the training Dataframe & validation DataFrame
    validation_df = relevant_pair_df.iloc[validation_indices]
    train_df = relevant_pair_df.drop(validation_indices)
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)

    # Load the query data
    query_df = pd.read_csv("/home/jhju/datasets/pdsearch/qid2query.tsv", sep='\t', header=None)
    query_df.columns = ["qid", "query"]

    # Merging the train_df with query_df
    train_df = pd.merge(train_df, query_df, on='qid', how='left')

    # Merging the validation_df with query_df
    validation_df = pd.merge(validation_df, query_df, on='qid', how='left')

    # Creating image paths for the train_df
    train_df['image_path'] = IMG_FILE_PATH + train_df['product'].astype(str) + '.jpg'

    # Creating image paths for the validation_df
    validation_df['image_path'] = IMG_FILE_PATH + validation_df['product'].astype(str) + '.jpg'
    
    train_df.rename(columns={'image_path': 'image', 'query': 'text'}, inplace=True)
    validation_df.rename(columns={'image_path': 'image', 'query': 'text'}, inplace=True)
    
    train_df = train_df.drop(columns=["qid", "product"]).reset_index(drop=True)
    validation_df = validation_df.drop(columns=["qid", "product"]).reset_index(drop=True)
    
    # load pandas dataframe to Dataset class
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    
    # Applying the transformation function to the datasets
    train_dataset.set_transform(transforms)
    validation_dataset.set_transform(transforms)
    
    print("finished preparing data")
    
    return train_dataset, validation_dataset


wer = load("wer")
rouge = load('rouge')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits[0]
    decoded_labels = PROCESSOR.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = PROCESSOR.batch_decode(predicted, skip_special_tokens=True)
    
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    rouge_scores = rouge.compute(predictions=decoded_predictions, references=decoded_labels)
    
    results = {'wer': wer_score}
    results.update(rouge_scores)
    
    return results


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# training
TRAINING_ARGS = TrainingArguments(
        output_dir="/tmp2/chiuws/fine_tuned_GIT/GIT-large-coco_pds",
        logging_dir="/tmp2/chiuws/fine_tuned_GIT/tensorboard_logs",
        report_to="tensorboard",
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type=SchedulerType.COSINE,
        num_train_epochs=5,
        fp16=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        save_total_limit=None,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=100,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=True,
    )

def train(training_data, validation_data):
    global MODEL, TRAINING_ARGS
    
    trainer = Trainer(
    model=MODEL,
    args=TRAINING_ARGS,
    train_dataset=training_data,
    eval_dataset=validation_data,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()

if __name__ == "__main__":
    training_data, validation_data = prepare_data()

    train(training_data, validation_data)
