import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

# 1. SETUP & CONFIGURATION
model_checkpoint = "microsoft/deberta-v3-base" 
model_dir = "./best_rte_model"
label_map = {"not_entailment": 0, "entailment": 1}
reverse_label_map = {0: "not_entailment", 1: "entailment"}

print("Loading and splitting the dataset...")
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# Strict 80/20 train/validation split
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Training set size: {len(split_dataset['train'])}")
print(f"Validation set size: {len(split_dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs["label"] = [label_map[label] for label in examples["label"]]
    return inputs

print("Tokenising the data...")
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

# 2. MODEL SETUP & LORA CONFIGURATION
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=reverse_label_map,
    label2id=label_map
)

# Expanded LoRA target modules for >85% logical reasoning accuracy
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query_proj", "key_proj", "value_proj", "dense"], 
    modules_to_save=["pooler", "classifier"] # FIX: Both heads are now saved
)

lora_model = get_peft_model(base_model, peft_config)

print("\n--- Model Parameters for your Model Card ---")
lora_model.print_trainable_parameters()
print("--------------------------------------------\n")

# 3. TRAINING LOOP
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./rte_lora_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,                  # Slightly higher LR since we have more parameters
    per_device_train_batch_size=2,       # 4GB VRAM limit
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,       # Effective batch size of 16
    num_train_epochs=15,                 # Give it time to hit 85%+
    weight_decay=0.01,
    warmup_steps=150,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

print("Starting training...")
trainer.train()

trainer.model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model and tokenizer successfully saved to {model_dir}")

# 4. INFERENCE ON VALIDATION DATA
print("\nGenerating predictions on the 20% validation split...")
val_predictions_output = trainer.predict(tokenized_datasets["test"])
val_predicted_classes = np.argmax(val_predictions_output.predictions, axis=-1)
val_predicted_labels = [reverse_label_map[pred] for pred in val_predicted_classes]

val_data = split_dataset["test"]
val_results_df = pd.DataFrame({
    "idx": val_data["idx"],
    "premise": val_data["premise"],
    "hypothesis": val_data["hypothesis"],
    "true_label": val_data["label"],
    "predicted_label": val_predicted_labels
})

val_results_df.to_csv("val_predictions.csv", index=False)

val_accuracy = val_predictions_output.metrics["test_accuracy"]
val_mistakes = val_results_df[val_results_df["true_label"] != val_results_df["predicted_label"]]

print(f"\n--- VALIDATION RESULTS (20% Split) ---")
print(f"Validation Accuracy: {val_accuracy:.4f} ({(val_accuracy * 100):.2f}%)")
print("Saved full validation breakdown to val_predictions.csv")