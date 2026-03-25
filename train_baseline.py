import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from utils import get_dataset

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "bert-base-uncased"
DATASET = ("tweet_eval", "sentiment")
OUTPUT_DIR = "./results/baseline"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
SAVE_STEPS = 500
LOGGING_STEPS = 100

print("🚀 Starting baseline training from scratch...")

# -------------------------
# DATASET
# -------------------------
ds = get_dataset(*DATASET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

ds = ds.map(preprocess, batched=True)
num_labels = len(set(ds["train"]["label"]))
print(f"✅ Dataset loaded and tokenized. Number of labels: {num_labels}")

# -------------------------
# MODEL
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
print("✅ Model loaded from scratch")

# -------------------------
# METRICS
# -------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

print("✅ Metrics loaded")

# -------------------------
# TRAINING ARGUMENTS
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,        # safe fresh start
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    save_total_limit=5,               # keep last 5 checkpoints
    fp16=False,                       # stable
    report_to="wandb",                # optional: W&B logging
    dataloader_pin_memory=False,      # avoids CUDA pinning errors
)

# -------------------------
# TRAINER
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("✅ Trainer initialized")

# -------------------------
# START TRAINING
# -------------------------
trainer.train()  # 🚀 safe fresh run
print("✅ Training complete")

# -------------------------
# EVALUATE
# -------------------------
results = trainer.evaluate(ds["test"])
print("✅ Test set evaluation:")
print(results)
