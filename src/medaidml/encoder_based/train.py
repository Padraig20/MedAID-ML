import os
import wandb
import time
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, IntervalStrategy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score
from datasets import Dataset
from typing import Tuple

from medaidml.utils import split_val_test, json_to_dataframe, \
    get_necessary_columns, convert_to_huggingface_dataset, \
    get_dataset_dict
from medaidml import DATA_TEST_JSON, DATA_TRAIN_JSON, LABEL_TO_ID
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Train a model using encoder-based methods.")
    parser.add_argument(
        "--model",
        type=str,
        default="google-bert/bert-base-multilingual-cased",
        choices=["microsoft/mdeberta-v3-base",
                 "FacebookAI/xlm-roberta-base",
                 "distilbert/distilbert-base-multilingual-cased",
                 "google-bert/bert-base-multilingual-cased"],
        help="Pre-trained model name or path to local checkpoint."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model and tokenizer."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the model."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Number of steps between evaluations."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer."
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use a small subset of the data for development."
    )
    return parser.parse_args()

def compute_metrics(pred: dict) -> dict:
    y_true = pred.label_ids                 # gold labels
    y_pred = pred.predictions.argmax(-1)    # predictions
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def tokenize(examples: dict) -> dict:
    return tokenizer(examples["text"], padding="max_length", max_length=MAX_LENGTH, truncation=True)

def get_prediction(model: AutoModelForSequenceClassification,
                   tokenizer: AutoTokenizer,
                   text: str) -> int:
    inputs = tokenizer(text, padding="max_length", max_length=MAX_LENGTH, truncation= True, return_tensors="pt").to(DEVICE)
    pred = model(**inputs).logits
    probs = pred.softmax(1)
    return probs.argmax().item()

def evaluate_model(model: AutoModelForSequenceClassification,
                   dataset: Dataset) -> pd.DataFrame:
    model.eval()
    y_true = dataset['label']
    texts = dataset['text']

    y_pred = [get_prediction(model, tokenizer, text) for text in texts]
    
    dataset_df = pd.DataFrame(dataset)[["text", "language", "label", "source"]]
    dataset_df['Ground Truth'] = y_true
    dataset_df['Prediction'] = y_pred
        
    return dataset_df

def evaluate_results(results_df: pd.DataFrame) -> None:
    print("Classification Report:")
    print(classification_report(results_df['Ground Truth'], results_df['Prediction'], target_names=LABEL_TO_ID))
    print(f"F1 Score: {f1_score(results_df['Ground Truth'], results_df['Prediction'], average='macro')}")
    print(f"Accuracy: {accuracy_score(results_df['Ground Truth'], results_df['Prediction'])}")

if __name__ == "__main__":
    args = get_args()
    
    MODEL = args.model
    MAX_LENGTH = args.max_length
    OUTPUT_DIR = args.output_dir
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    EVAL_STEPS = args.eval_steps
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    SEED = args.seed
    DEVELOPMENT = args.development
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data = json_to_dataframe(DATA_TRAIN_JSON)
    no_dataleak_test_df = json_to_dataframe(DATA_TEST_JSON)
    
    train_df, val_df, test_df = split_val_test(train_data, seed=SEED)
    
    train_df = get_necessary_columns(train_df)
    val_df = get_necessary_columns(val_df)
    test_df = get_necessary_columns(test_df)
    no_dataleak_test_df = get_necessary_columns(no_dataleak_test_df)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"No dataleak test data shape: {no_dataleak_test_df.shape}")
    print(f"Model: {MODEL}")
    
    train_dataset = convert_to_huggingface_dataset(train_df)
    val_dataset = convert_to_huggingface_dataset(val_df)
    test_dataset = convert_to_huggingface_dataset(test_df)
    no_dataleak_test_dataset = convert_to_huggingface_dataset(no_dataleak_test_df)
    
    final_dataset = get_dataset_dict(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        no_dataleak_test_dataset=no_dataleak_test_dataset
    )
    
    if DEVELOPMENT:
        final_dataset['train'] = final_dataset['train'].shuffle(seed=SEED).select(range(100))
        final_dataset['validation'] = final_dataset['validation'].shuffle(seed=SEED).select(range(20))
        final_dataset['test'] = final_dataset['test'].shuffle(seed=SEED).select(range(20))
        final_dataset['no_dataleak_test'] = final_dataset['no_dataleak_test'].shuffle(seed=SEED).select(range(20))
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABEL_TO_ID)).to(DEVICE)

    encoded_data = final_dataset.map(tokenize, batched=True)

    # params for training
    args = TrainingArguments(output_dir=OUTPUT_DIR)
    args.num_train_epochs = NUM_EPOCHS
    args.per_device_train_batch_size = BATCH_SIZE
    args.per_device_eval_batch_size = BATCH_SIZE
    args.evaluation_strategy = IntervalStrategy.STEPS
    args.eval_strategy = IntervalStrategy.STEPS
    args.eval_steps = EVAL_STEPS
    args.metric_for_best_model = 'eval_loss'
    args.load_best_model_at_end=True
    args.learning_rate=LEARNING_RATE
    args.weight_decay=WEIGHT_DECAY
    args.logging_steps=EVAL_STEPS    

    ## training
    
    torch.cuda.empty_cache()

    wandb.init(project="medaidml")

    trainer = Trainer(
        model = model,
        train_dataset = encoded_data['train'],
        eval_dataset = encoded_data['validation'],
        args = args,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time

    print(MODEL, " has finished training...")
    print(f"Training time: {training_time:.2f} seconds")

    wandb.finish() # finalize wandb run

    ## evaluate test set

    print("Evaluating on test set...")

    results_test_df = evaluate_model(model, encoded_data['test'])
    evaluate_results(results_test_df)
    
    ## evaluate no dataleak test set
    print("Evaluating on no dataleak test set...")
    results_no_dataleak_df = evaluate_model(model, encoded_data['no_dataleak_test'])
    evaluate_results(results_no_dataleak_df)
    
    ## save evaluated results
    results_test_df.to_csv(os.path.join(OUTPUT_DIR, "results_test.csv"), index=False)
    results_no_dataleak_df.to_csv(os.path.join(OUTPUT_DIR, "results_no_dataleak.csv"), index=False)

    print("Done!")