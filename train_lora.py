import time, os
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# -------------------------
# CONFIG
# -------------------------
MODEL = "bert-base-uncased"
DATASET = ("tweet_eval", "sentiment")
RANK = int(os.environ.get("LORA_RANK", 8))  # set RANK via env
OUTPUT_DIR = f"./results/lora_r{RANK}"
BATCH = 16
EPOCHS = 3
LR = 2e-4
SEED = 42
MAX_LEN = 128

# -------------------------
# DATASET
# -------------------------
ds = load_dataset(*DATASET)
tok = AutoTokenizer.from_pretrained(MODEL)

if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "[PAD]"})

def preprocess(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

ds = ds.map(preprocess, batched=True)
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
num_labels = len(set(ds["train"]["labels"]))

# -------------------------
# MODEL + LoRA
# -------------------------
base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
config = LoraConfig(r=RANK, lora_alpha=16, target_modules=["query", "value"], task_type=TaskType.SEQ_CLS)
model = get_peft_model(base, config)

if len(tok) != model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(len(tok))

# -------------------------
# METRICS
# -------------------------
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# -------------------------
# TRAINING
# -------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics
)

torch.cuda.reset_peak_memory_stats()
start = time.time()
trainer.train()
train_time = time.time() - start

eval_metrics = trainer.evaluate(ds["test"])
print("Test Metrics:", eval_metrics)

# -------------------------
# STATS + SAVE
# -------------------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
peak_mem = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else None

summary = {
    "strategy": "LoRA",
    "rank": RANK,
    "trainable_params": int(trainable),
    "peak_gpu_mem_mb": peak_mem,
    "train_time_s": train_time,
    "accuracy": eval_metrics["eval_accuracy"],
    "f1": eval_metrics["eval_f1"]
}
pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
print("Saved:", os.path.join(OUTPUT_DIR, "summary.csv"))
