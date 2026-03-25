import torch
import transformers
import datasets
import pandas as pd
import sklearn
import bitsandbytes as bnb
import peft
import accelerate
import evaluate

print("="*60)
print("✅ Environment Check Report")
print("="*60)

# Torch / CUDA
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA version (from PyTorch): {torch.version.cuda}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
else:
    print("⚠️ No GPU detected. Training will be very slow!")

# Hugging Face
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"Evaluate: {evaluate.__version__}")

# Utilities
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"bitsandbytes: {bnb.__version__}")

print("="*60)
print("✅ All key packages imported successfully!")
print("="*60)
