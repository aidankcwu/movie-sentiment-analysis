import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate


# Config
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = "data/imdb_binary"
OUT_DIR = "models/distilbert_imdb_binary"
MAX_LENGTH = 256


def tokenize_text(batch, tokenizer):
    """Tokenize review text with truncation"""
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


def compute_metrics(eval_pred, accuracy_metric, f1_metric):
    """Calculate accuracy and F1 score from predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
    
    return {"accuracy": acc, "f1": f1}


def main():
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load CSV splits
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"{DATA_DIR}/train.csv",
            "validation": f"{DATA_DIR}/val.csv",
            "test": f"{DATA_DIR}/test.csv",
        },
    )
    
    print(f"Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,} | Test: {len(dataset['test']):,}")
    
    # Load tokenizer and tokenize text
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_dataset = dataset.map(
        lambda batch: tokenize_text(batch, tokenizer),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "label"],
    )
    
    # Setup model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Data collator handles padding dynamically
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Training hyperparameters
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.06,
        
        # Performance
        fp16 = False,  # set to False if you don't have GPU or get errors
        report_to="none",
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, accuracy_metric, f1_metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("starting training")
    trainer.train()
    
    # Evaluate on validation set
    print("\nValidation metrics:")
    val_metrics = trainer.evaluate()
    print(val_metrics)
    
    # Evaluate on test set
    print("\nTest metrics:")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    print(test_metrics)
    
    # Save model and tokenizer
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()