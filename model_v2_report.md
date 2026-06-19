# Model v2 — Multi-class Email Threat Classifier

## Overview

- **Model**: LinearSVC + CalibratedClassifierCV
- **Pipeline**: TextPreprocessor -> TF-IDF (30k features, 1-2 ngrams) -> LinearSVC + CalibratedClassifierCV
- **Classes**: ham, spam, phishing
- **Split**: train=23,818 / val=2,977 / test=2,978 (80/10/10, stratified)

## Training Class Balance

| Class | Count | % |
|---|---|---|
| ham | 10,454 | 43.9% |
| spam | 8,137 | 34.2% |
| phishing | 5,227 | 21.9% |

## Test Set Results

**Overall accuracy: 0.9251**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ham | 0.9705 | 0.9816 | 0.9760 | 1307 |
| spam | 0.9174 | 0.8850 | 0.9009 | 1017 |
| phishing | 0.8474 | 0.8746 | 0.8608 | 654 |

## Confusion Matrix

Rows = actual, columns = predicted.

| | ham | spam | phishing |
|---|---|---|---|
| **ham** | 1283 | 15 | 9 |
| **spam** | 23 | 900 | 94 |
| **phishing** | 16 | 66 | 572 |

## Phishing Focus

- **Precision**: 0.8474 (of emails flagged phishing, 84.7% really are)
- **Recall**: 0.8746 (of real phishing emails, 87.5% are caught)
- **F1**: 0.8608
