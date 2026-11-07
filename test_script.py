import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import PeftModel

# 1. SETUP
model_checkpoint = "microsoft/deberta-v3-base" 
model_dir = "./best_rte_model"
test_file = "test_2.jsonl" 

label_map = {"not_entailment": 0, "entailment": 1}
reverse_label_map = {0: "not_entailment", 1: "entailment"}

print(f"Loading test data from {test_file}...")
try:
    test_dataset = load_dataset("json", data_files=test_file, split="train")
except FileNotFoundError:
    print(f"Error: Could not find {test_file}.")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model_dir)

def preprocess_function(examples):
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    if "label" in examples:
        inputs["label"] = [label_map[label] for label in examples["label"]]
    return inputs

tokenized_test = test_dataset.map(preprocess_function, batched=True)

# 2. LOAD MODEL SAFELY
print("Loading base model and injecting trained LoRA weights...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=reverse_label_map,
    label2id=label_map
)

model = PeftModel.from_pretrained(base_model, model_dir)

# 3. RUN INFERENCE
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

inference_trainer = Trainer(
    model=model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Running predictions...")
predictions_output = inference_trainer.predict(tokenized_test)

predicted_classes = np.argmax(predictions_output.predictions, axis=-1)
predicted_labels = [reverse_label_map[pred] for pred in predicted_classes]

# 4. SAVE RESULTS
results_dict = {
    "idx": test_dataset["idx"],
    "premise": test_dataset["premise"],
    "hypothesis": test_dataset["hypothesis"],
    "predicted_label": predicted_labels
}

if "label" in test_dataset.column_names:
    results_dict["true_label"] = test_dataset["label"]
    
results_df = pd.DataFrame(results_dict)
results_df.to_csv("final_test_predictions.csv", index=False)

print("\n--- FINAL TEST RESULTS ---")
if "label" in test_dataset.column_names:
    accuracy = predictions_output.metrics.get("test_accuracy", 0.0)
    print(f"Test Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")
print("Saved full breakdown to final_test_predictions.csv")