import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ------------------------- CONFIG -------------------------
MODEL = "roberta-base"
DATASET = ("tweet_eval", "sentiment")
RANK = 8           # Change to 4, 8, 16
OUTPUT_DIR = f"./results/qlora_r{RANK}"
BATCH = 16
EPOCHS = 3
LR = 2e-4
MAX_LEN = 128

# ------------------------- DATASET -------------------------
print("📥 Loading dataset...")
ds = load_dataset(*DATASET)
tok = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

ds = ds.map(preprocess, batched=True)
num_labels = len(set(ds["train"]["label"]))

# ------------------------- QUANTIZATION -------------------------
print("⚙️ Setting up quantization...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

print("⚙️ Loading model with QLoRA...")
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels,
    quantization_config=bnb,
    device_map="auto"
)

# ------------------------- LoRA -------------------------
print("⚙️ Adding LoRA adapters...")
lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=16,
    target_modules=["query", "value"],  # Works for Roberta
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(base, lora_cfg)

# ------------------------- TRAINER -------------------------
print("⚙️ Preparing Trainer...")
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    compute_metrics=lambda p: {
        "accuracy": (np.argmax(p.predictions, -1) == p.label_ids).mean()
    }
)

# ------------------------- RUN -------------------------
print("🔍 Checking device...")
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Running on:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No GPU detected! Training may be very slow or fail with QLoRA.")

print("🚀 Starting training...")
train_result = trainer.train()
print("✅ Training finished!")

print("📊 Evaluating on test set...")
metrics = trainer.evaluate(ds["test"])
print("Test metrics:", metrics)
