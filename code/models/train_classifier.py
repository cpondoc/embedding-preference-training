"""
Code to train a binary classifier. Adapted from: https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier/blob/main/src/train_edu_bert.py.
"""

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, concatenate_datasets, ClassLabel
import numpy as np
import evaluate
import os
from sklearn.metrics import classification_report, confusion_matrix

"""
Constants for the dataset.
"""
DATASET_PATH = "data/wiki-vs/"
BASE_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
CHECKPOINT_DIR = "scratch/sample-run/"


def compute_metrics(eval_pred):
    """
    Compute metrics such as precision, recall, and accuracy.
    """
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main():
    """
    Load in the datasets and perform training.
    """
    # Paths to each class folder
    good_path = DATASET_PATH + "good/*"
    bad_path = DATASET_PATH + "bad/*"

    # Load the "good" and "bad" folders as separate datasets
    good_dataset = load_dataset(
        "text",
        data_files={"train": good_path},
        split="train",
        cache_dir="scratch/cosmo/cache/",
    ).map(
        lambda _: {"label": 1}, num_proc=8
    )  # Label "good" as 1

    bad_dataset = load_dataset(
        "text",
        data_files={"train": bad_path},
        split="train",
        cache_dir="scratch/cosmo/cache/",
    ).map(
        lambda _: {"label": 0}, num_proc=8
    )  # Label "bad" as 0

    # Concatenate the "good" and "bad" datasets (only selecting 15k from each dataset right now)
    good_sample = good_dataset.shuffle(seed=42).select(range(15000))
    bad_sample = bad_dataset.shuffle(seed=42).select(range(15000))
    dataset = concatenate_datasets([good_sample, bad_sample])

    # Shuffle the dataset (optional, if desired for training)
    dataset = dataset.shuffle(seed=42)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True, padding=True)
        batch["labels"] = examples["label"]
        return batch

    # Preprocess each of the datasets
    train_dataset = train_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0,
    )
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        learning_rate=3e-4,
        num_train_epochs=10,
        seed=0,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))


if __name__ == "__main__":
    main()
