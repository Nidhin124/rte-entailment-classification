---
base_model: microsoft/deberta-v3-base
library_name: peft
tags:
- base_model:adapter:microsoft/deberta-v3-base
- lora
- transformers
- rte
- nli
- sequence-classification
---

# Model Card for RTE LoRA Adapter

This is a Parameter-Efficient Fine-Tuning (PEFT) adapter built on top of `microsoft/deberta-v3-base`, designed specifically for the Recognizing Textual Entailment (RTE) challenge. It uses Low-Rank Adaptation (LoRA) to achieve high accuracy while operating strictly within a 4GB VRAM hardware constraint.

## Model Details

### Model Description

This model classifies sentence pairs (premise and hypothesis) into two categories: `entailment` and `not_entailment`. Instead of updating all 184M base parameters, it trains lightweight LoRA matrices injected into the attention and dense layers, while completely unfreezing the `pooler` and `classifier` heads to accurately capture complex logical reasoning.

- **Developed by:** Nidhin (M.Tech, NIT Calicut)
- **Model type:** Sequence Classification (LoRA Adapter)
- **Language(s) (NLP):** English
- **License:** MIT / Academic Use
- **Finetuned from model:** `microsoft/deberta-v3-base`

### Model Sources 

- **Repository:** Local submission for CS6315E

## Uses

### Direct Use

The model is intended to be used for the CS6315E academic evaluation to predict textual entailment on unseen test sets (`test.jsonl`). It is strictly an adapter and must be loaded alongside the base `microsoft/deberta-v3-base` model.

### Out-of-Scope Use

This model is fine-tuned specifically on a constrained academic dataset. It is not intended for production-grade legal, medical, or financial document entailment without further domain-specific training.

## Bias, Risks, and Limitations

Because the model is built on DeBERTa-v3, it inherits the biases of the base model's pre-training data. In the context of RTE, it may occasionally struggle with highly complex double negations or domains heavily reliant on external world knowledge not present in the training set.

### Recommendations

Users should ensure that the base model and tokenizer are instantiated with the exact same label mappings (`{"not_entailment": 0, "entailment": 1}`) used during training to prevent inverted predictions.

## How to Get Started with the Model

Use the following code to run inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

model_checkpoint = "microsoft/deberta-v3-base"
adapter_dir = "./best_rte_model"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
