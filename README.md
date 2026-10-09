# Hybrid AI Content Moderation

A BERT-based multi-label toxicity classification module developed as part of a **Hybrid AI Framework for Automated Content Moderation**.

The system identifies different types of harmful content and provides probability-based predictions that can later be used by severity scoring and decision-making modules.

---

## Project Status

### Completed

- Jigsaw Toxic Comment dataset preprocessing
- Text cleaning and normalization
- BERT tokenizer integration
- Multi-label BERT classification
- Class-weighted training
- Multi-label stratified train/validation/test splitting
- Validation-based threshold optimization
- Per-class Precision, Recall, and F1 evaluation
- Micro F1 and Macro F1 evaluation
- Interactive inference
- Model training and saving
- 20K-sample experiment

### Planned Next

- Random Forest severity scorer
- Severity score generation
- Decision engine
- Allow / Warn / Remove recommendations
- Human-in-the-loop moderation
- Moderation dashboard and audit system

---

# Architecture

## Current BERT Module

```text
Jigsaw Toxic Comment Dataset
            |
            v
     Data Preprocessing
            |
            v
 Multi-label Stratified Split
            |
            v
       BERT Tokenizer
            |
            v
      BERT Fine-Tuning
            |
            v
     Class-Weighted Loss
            |
            v
  Six Toxicity Categories
            |
            v
 Validation Threshold Tuning
            |
            v
   Final Test Evaluation
```

## Planned Complete System

```text
                    User Comment
                         |
                         v
                  BERT Classifier
                         |
            +------------+------------+
            |                         |
            v                         v
     Violation Types          Confidence Scores
            |                         |
            +------------+------------+
                         |
                         v
                  Severity Scorer
                  (Random Forest)
                         |
                         v
                  Decision Engine
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
           ALLOW       WARN       REMOVE
                                     |
                                     v
                               Human Review
```

---

# Dataset

The project uses the **Jigsaw Toxic Comment Classification Dataset**.

Original dataset size:

```text
159,563 comments
```

For the current experiment, a subset of:

```text
20,000 comments
```

was used.

## Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Testing | 2,000 |
| Total | 20,000 |

The six toxicity labels are:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The dataset itself is not included in this repository because of its size.

---

# Model

## BERT

The project uses:

```text
bert-base-uncased
```

with a multi-label classification head.

A comment can belong to multiple toxicity categories simultaneously.

For example:

```text
Input:
You are an idiot and I hate you
```

may produce predictions for multiple categories such as:

```text
toxic
obscene
insult
```

Each category receives an independent probability.

---

# Training Configuration

| Parameter | Value |
|---|---|
| Base Model | `bert-base-uncased` |
| Dataset | Jigsaw Toxic Comments |
| Experiment Size | 20,000 |
| Training Samples | 16,000 |
| Validation Samples | 2,000 |
| Test Samples | 2,000 |
| Epochs | 1 |
| Maximum Sequence Length | 64 |
| Training Batch Size | 4 |
| Evaluation Batch Size | 4 |
| Gradient Accumulation | 4 |
| Learning Rate | `2e-5` |
| Weight Decay | `0.01` |
| Loss | Class-weighted BCE |
| Device | CPU |

---

# Class-Weighted Training

The dataset is highly imbalanced, particularly for categories such as:

- `threat`
- `severe_toxic`
- `identity_hate`

To reduce the effect of class imbalance, the model uses a **weighted Binary Cross-Entropy loss**.

The objective is to prevent the model from learning only the majority classes while ignoring minority categories.

---

# Threshold Optimization

Instead of assuming:

```text
Probability >= 0.5 → Positive
Probability < 0.5  → Negative
```

the project determines a separate classification threshold for each label using the **validation dataset**.

The optimized thresholds are then applied to the untouched test set.

This prevents the test set from being used to select the thresholds.

---

# Results

## 20K Experiment

Final test performance:

| Metric | Score |
|---|---:|
| **Micro F1** | **0.6922** |
| **Macro F1** | **0.5029** |

## Per-Class Results

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| Toxic | 0.7877 | 0.7698 | **0.7787** |
| Severe Toxic | 0.3265 | 0.9677 | **0.4898** |
| Obscene | 0.6716 | 0.8879 | **0.7631** |
| Threat | 0.0000 | 0.0000 | **0.0000** |
| Insult | 0.5743 | 0.8614 | **0.6905** |
| Identity Hate | 0.2372 | 0.3929 | **0.2955** |

> The `threat` category remains challenging because it contains substantially fewer positive examples than the major toxicity categories.

---

# Example Inference

Run:

```bash
cd src
python inference.py
```

Example:

```text
AI-Powered Content Moderation System
====================================

Enter comment: You are an idiot and I hate you
```

The model returns probabilities for:

```text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

Example output format:

```text
toxic           | Probability: 0.6974 | YES
severe_toxic    | Probability: 0.2727 | NO
obscene         | Probability: 0.5610 | YES
threat          | Probability: 0.2926 | NO
insult          | Probability: 0.4755 | NO
identity_hate   | Probability: 0.2493 | NO
```

---

# Project Structure

```text
Hybrid-AI-Content-Moderation/
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── threshold_tuning.py
│   └── train.py
│
├── data/
│   └── train.csv              # Not included in Git
│
├── models/
│   └── ...                    # Not included in Git
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/chaitu4206/Hybrid-AI-Content-Moderation.git
```

Move into the project:

```bash
cd Hybrid-AI-Content-Moderation
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- Hugging Face Transformers
- Accelerate
- Pandas
- NumPy
- Scikit-learn
- Iterative Stratification

---

# Dataset Setup

The dataset is intentionally excluded from GitHub.

Place the Jigsaw dataset at:

```text
data/train.csv
```

Expected columns:

```text
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

---

# Training

From the project root:

```bash
cd src
python train.py
```

The training script performs:

1. Dataset loading
2. Text preprocessing
3. Multi-label stratified splitting
4. Tokenization
5. Class-weight calculation
6. BERT fine-tuning
7. Validation evaluation
8. Threshold optimization
9. Test evaluation
10. Model saving

---

# Evaluation

To evaluate a saved BERT model:

```bash
cd src
python evaluate.py
```

---

# Inference

To test the trained model interactively:

```bash
cd src
python inference.py
```

Enter a comment when prompted.

The model returns:

- Probability for each toxicity category
- Binary toxicity prediction

---

# Threshold Tuning

The project also contains:

```text
src/threshold_tuning.py
```

This script can be used for experimental threshold analysis.

For the final evaluation pipeline, thresholds should be selected using validation data and then applied to the untouched test set.

---

# Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **BERT**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Iterative Stratification**

---

# Limitations

The current model has several limitations:

1. Training was performed on CPU.
2. Only 20,000 of the original 159,563 comments were used for the current experiment.
3. The dataset is highly imbalanced.
4. The `threat` category remains difficult to classify.
5. The current system is focused on toxicity classification and does not yet implement the complete severity/decision pipeline.
6. The trained model files are not stored in this GitHub repository.

---

# Future Work

The next stage of the project will extend the BERT classifier into a complete hybrid moderation framework.

## 1. Severity Scoring

Develop a Random Forest model that combines:

- BERT violation categories
- BERT confidence scores
- Number of detected violations
- Severity-related features
- Report patterns

to generate a severity score.

## 2. Decision Engine

The decision engine will combine:

- BERT predictions
- Severity score
- Confidence
- Rule-based conditions

to recommend:

```text
ALLOW
WARN
REMOVE
ESCALATE TO HUMAN REVIEW
```

## 3. Human-in-the-Loop

Low-confidence or high-risk content can be forwarded to a human moderator for review.

## 4. Moderation Dashboard

A future dashboard can provide:

- Reported comments
- Toxicity categories
- Confidence scores
- Severity levels
- Recommended actions
- Moderator decisions
- Audit history

---

# Project Pipeline

```text
User-Reported Content
        |
        v
Data Preprocessing
        |
        v
BERT Toxicity Classification
        |
        v
Violation Categories + Confidence
        |
        v
Random Forest Severity Scoring
        |
        v
Decision Engine
        |
        +------------+------------+-------------+
        |            |            |             |
        v            v            v             v
      ALLOW        WARN        REMOVE     HUMAN REVIEW
```

---

# Current Development Status

```text
[✓] Dataset preprocessing
[✓] BERT model implementation
[✓] Multi-label classification
[✓] Class-weighted training
[✓] Stratified data splitting
[✓] Threshold optimization
[✓] Model evaluation
[✓] Inference
[✓] 20K experiment
[✓] GitHub repository

[ ] Random Forest severity scorer
[ ] Decision engine
[ ] Human-in-the-loop module
[ ] Moderation dashboard
[ ] Audit system
```

---

# Author

**Chaitanya Swarup Reddy Pathakota**

GitHub: https://github.com/chaitu4206

---

## License

This project is developed for academic and research purposes.
