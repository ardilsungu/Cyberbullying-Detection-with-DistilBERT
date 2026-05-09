# 🛡️ Cyberbullying Detection — Advanced NLP & DistilBERT

A research-grade NLP project that detects and classifies cyberbullying types in English tweets using classical ML models and a fine-tuned DistilBERT transformer — trained on GPU.

---

## 📊 Results at a Glance

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 82.11% | 0.82 |
| SVM — LinearSVC (TF-IDF) | 81.41% | 0.81 |
| **DistilBERT (fine-tuned)** | **86%** ✅ | **0.86** |

**Dataset:** 47,692 tweets · **Classes:** 6 · **Split:** 70 / 15 / 15

---

## 🏷️ Classes

| Label | Description |
|---|---|
| `age` | Age-based cyberbullying |
| `ethnicity` | Ethnicity-based cyberbullying |
| `gender` | Gender-based cyberbullying |
| `religion` | Religion-based cyberbullying |
| `other_cyberbullying` | Other forms of cyberbullying |
| `not_cyberbullying` | Not cyberbullying |

---

## ✨ What Makes This Project Different

### 1. Advanced Text Preprocessing Pipeline
Most projects just strip punctuation. This pipeline handles social media reality:

- **Emoticon → Text:** `:)` → `smile`, `>:(` → `angry`
- **Slang expansion:** `kys` → `kill yourself`, `stfu` → `shut the fuck up` *(critical for detecting disguised bullying)*
- **Leetspeak / Obfuscation:** `f*ck` → `fuck`, `@` → `a` *(catches censored hate speech)*
- **Cashtag/Hashtag splitting:** `#ImACelebrity` → `Im A Celebrity`
- **Contraction expansion:** `don't` → `do not`, `I'm` → `I am`
- **Lemmatization** with negation preservation (`not`, `never`, `nor` are kept)
- **Turkish character normalization** for mixed-language datasets

### 2. Three-Way Model Comparison
Classical vs. deep learning — not just training one model and calling it done.

### 3. GPU-Accelerated DistilBERT Fine-Tuning
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU
- 3 epochs with warmup, weight decay, fp16 precision
- Best model checkpoint saved via `load_best_model_at_end=True`

---

## 🏆 Best Model — DistilBERT Detail

```
                     precision  recall  f1-score  support
age                    0.97      0.98      0.97     1199
ethnicity              0.98      0.99      0.98     1194
gender                 0.89      0.90      0.90     1193
not_cyberbullying      0.69      0.59      0.63     1179
other_cyberbullying    0.68      0.76      0.72     1149
religion               0.96      0.96      0.96     1200

accuracy                                  0.86     7114
```

> **Note:** `not_cyberbullying` and `other_cyberbullying` are the hardest classes — subtle language makes them difficult to separate even for transformers.

---

## ⚙️ Pipeline Overview

```
Raw Tweets (47,692)
        │
        ▼
Advanced Preprocessing
  ├─ Emoticon / Slang / Leetspeak dictionaries
  ├─ URL, mention, RT removal
  ├─ Contraction expansion
  ├─ Lemmatization (WordNetLemmatizer)
  └─ Selective stopword removal (negations preserved)
        │
        ▼
  ┌─────┴──────┐
  │            │
TF-IDF      DistilBERT
(1+2 grams)  Tokenizer
  │            │
  ├─ LogReg   Fine-tune
  └─ SVM      (3 epochs, GPU)
        │
        ▼
Evaluation: Accuracy · F1 · Confusion Matrix
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install pandas scikit-learn nltk matplotlib seaborn wordcloud emoji
pip install torch transformers  # for DistilBERT
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

> ⚠️ DistilBERT training requires a CUDA-capable GPU. CPU training is possible but very slow.

### Run

```bash
jupyter notebook projectt.ipynb
```

> Make sure `cyberbullying_tweets.csv` is in the same directory.

---

## 📁 Project Structure

```
cyberbullying_detection_v2/
├── cyberbullying_tweets.csv              # Raw dataset
├── cyberbullying_tweets_CLEANED.csv      # Preprocessed dataset
├── projectt.ipynb                        # Main notebook
├── final_cyberbullying_model_v3/         # Saved DistilBERT model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── README.md
```

---

## 🔍 Key Findings

- DistilBERT outperforms classical ML by ~4% — significant for a 6-class problem
- `ethnicity` and `age` are easiest to classify (F1 ≥ 0.97) across all models
- `not_cyberbullying` is the hardest class — ambiguous language causes ~40% misclassification
- Slang & obfuscation expansion in preprocessing meaningfully improves detection of disguised hate speech

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `DistilBERT` · `scikit-learn` · `NLTK` · `TF-IDF` · `WordCloud` · `Seaborn` · `CUDA`
