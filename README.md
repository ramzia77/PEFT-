

# 📘 Parameter-Efficient Fine-Tuning of LLMs: LoRA vs QLoRA

## 🔍 Overview
This project presents a comparative analysis of **Parameter-Efficient Fine-Tuning (PEFT)** methods—**LoRA** and **QLoRA**—against traditional full fine-tuning on a BERT-based model.

The goal is to evaluate how efficiently Large Language Models (LLMs) can be adapted to downstream tasks under **compute and memory constraints**.

---

## 🚀 Key Contributions
- Comparative study of **LoRA vs Full Fine-Tuning**
- Theoretical + empirical analysis of **QLoRA**
- Identification of **practical limitations and challenges**
- A **decision-making framework** for practitioners
- Discussion of emerging methods like **IR-QLoRA**

---

## ⚙️ Methods

### 1. Full Fine-Tuning (FT)
- Updates **all model parameters**
- High performance but **computationally expensive**

### 2. LoRA (Low-Rank Adaptation)
- Freezes base model weights
- Learns **low-rank matrices (A, B)** for updates
- Reduces trainable parameters drastically

### 3. QLoRA
- Extends LoRA with **4-bit quantization (NF4)**
- Enables training **very large models on limited hardware**

**Key features:**
- 4-bit NF4 quantization  
- Double quantization  
- Paged optimizers  

---

## 📊 Experimental Setup
- **Model:** `bert-base-uncased`
- **Dataset:** TweetEval (Sentiment)
- **Comparison:** Full Fine-Tuning vs LoRA (rank = 8)
- **Hardware:** RTX 4050 Laptop GPU

---

## 📈 Results

| Metric                | Full FT | LoRA |
|---------------------|--------|------|
| Peak Validation F1  | 74.6%  | 73.0% |
| Test F1             | 69.0%  | 68.6% |
| Trainable Params    | 100%   | <10% |
| Throughput          | 8.6    | 32 |
| Overfitting         | High   | Minimal |

### 🔑 Insights
- Full FT achieves slightly higher validation performance  
- **LoRA matches test performance** with:
  - ⚡ ~4× higher training speed  
  - 💾 Significant memory savings  
- LoRA acts as **implicit regularization**

---

## ⚖️ Trade-offs

### ✅ Advantages (LoRA / QLoRA)
- Significant **memory reduction**
- Faster training
- No inference latency
- Scalable to large models

### ⚠️ Limitations
- Structural differences → *“intruder dimensions”*
- Risk of **catastrophic forgetting**
- Less effective in **extremely low-data settings**
- QLoRA setup can be **fragile (dependency issues, OS constraints)**

---

## 🧠 When to Use What?

| Scenario                          | Recommended Approach |
|----------------------------------|---------------------|
| Limited compute                  | LoRA / QLoRA |
| Very large models (e.g., 65B)    | QLoRA |
| Extremely small datasets (<200)  | Few-shot prompting |
| Maximum peak performance         | Full Fine-Tuning |

---

## 🔮 Future Directions

### IR-QLoRA (Next-Gen PEFT)
Focuses on **information retention instead of just compression**.

**Key ideas:**
- Information Calibration Quantization (ICQ)
- Information Elastic Connection (IEC)

📈 Improves accuracy with **minimal overhead**

---

## 🧾 Conclusion
- **LoRA achieves near-equivalent performance** to full fine-tuning at a fraction of the cost  
- **QLoRA democratizes LLM fine-tuning** on limited hardware  
- Future methods focus on **information-aware adaptation**

---

## 📌 Takeaway
> Efficient fine-tuning is not just about reducing parameters—it’s about **preserving knowledge while adapting models effectively**.
