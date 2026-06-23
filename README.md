# AI Email Threat Scanner

A multi-class email threat classifier that detects **spam**, **phishing**, and **safe (ham)** emails using machine learning. Deployed as a full-stack web app with a FastAPI backend and a single-page frontend.

**Live demo:** [end-to-end-spam-classier.onrender.com](https://end-to-end-spam-classier.onrender.com)

> The server runs on Render's free tier. If the first request takes ~30 seconds, the container is waking up from idle. Hit `/health` to pre-warm it.

---

## How It Works

```
User pastes email text
        │
        ▼
┌───────────────────┐
│  TextPreprocessor  │  URLs → urltoken, money → moneytoken, ALL-CAPS → capstoken, etc.
└────────┬──────────┘
         ▼
┌───────────────────┐
│  TF-IDF Vectorizer │  30,000 features, unigrams + bigrams, sublinear TF
└────────┬──────────┘
         ▼
┌───────────────────┐
│  LinearSVC (SVM)   │  Calibrated probabilities via CalibratedClassifierCV
└────────┬──────────┘
         ▼
   ham / spam / phishing   +   confidence score   +   risk score   +   signal analysis
```

The model classifies emails into three categories and returns calibrated probability scores. The API layer adds signal analysis on top: URL pattern detection, urgency/financial-incentive heuristics, and TF-IDF-based phrase flagging.

---

## Model Performance

Trained on ~30,000 emails from 5 combined datasets. Evaluated on a held-out test set (10% of data, stratified).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Ham | 0.9705 | 0.9816 | 0.9760 |
| Spam | 0.9174 | 0.8850 | 0.9009 |
| Phishing | 0.8474 | 0.8746 | 0.8608 |

**Overall accuracy: 92.5%**

Phishing recall of 87.5% means the model catches ~7 out of 8 real phishing emails. See [model_v2_report.md](model_v2_report.md) for the full confusion matrix and training details.

---

## Training Datasets

| Source | Type | Rows Used |
|---|---|---|
| [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | SMS spam/ham | 5,085 |
| [Enron-Spam](https://github.com/MWiechmann/enron_spam_data) | Email spam/ham | 4,991 |
| [CEAS 2008](https://www.ceas.cc/) | Email spam/ham | 14,333 |
| [SpamAssassin](https://spamassassin.apache.org/publiccorpus/) | Email spam/ham | 5,253 |
| [Nazario Phishing Corpus](https://monkey.org/~jose/phishing/) | Phishing + safe email | 14,313 |

Each source was capped at 8,000 rows per class. Ham was further downsampled to 2x the phishing count to prevent class imbalance from drowning out the phishing class. `class_weight='balanced'` was applied on top.

---

## API

### `POST /predict`

```json
{
  "text": "Your PayPal account has been limited. Verify immediately...",
  "sender_domain": "paypal-secure-verify.com"
}
```

`sender_domain` is optional. If omitted, `sender_trust` returns `"Not available"`.

**Response:**

```json
{
  "prediction": "phishing",
  "confidence": 0.66,
  "risk_score": 100,
  "signals": {
    "urgent_language": "High",
    "suspicious_links": "Medium",
    "financial_incentive": "Low",
    "sender_trust": "Low"
  },
  "flagged_phrases": ["suspended", "verify", "click", "frozen"],
  "category_tags": ["Phishing"]
}
```

| Field | Source | Description |
|---|---|---|
| `prediction` | Model | Top predicted class: `ham`, `spam`, or `phishing` |
| `confidence` | Model | Calibrated probability of the predicted class (0–1) |
| `risk_score` | Model | `100 - (ham_probability * 100)`, clamped 0–100 |
| `flagged_phrases` | Model | Top TF-IDF x SVM-coefficient features for the predicted class |
| `signals.suspicious_links` | Regex | URL analysis: raw IPs, shorteners, brand mimicry, suspicious TLDs |
| `signals.urgent_language` | Heuristic | Curated urgency phrase matching (display layer only) |
| `signals.financial_incentive` | Heuristic | Hard/soft keyword split with co-occurrence gating |
| `signals.sender_trust` | Input | Domain reputation check if `sender_domain` provided |
| `category_tags` | Derived | Direct map from prediction |

### `GET /health`

Returns `{"status": "ok"}`. Does not load the model. Use this to pre-warm the Render container.

---

## Project Structure

```
├── main.py                    # FastAPI app — /predict, /health, signal analysis
├── preprocessing.py           # TextPreprocessor (sklearn transformer)
├── train_model_v2.py          # Training script for the v2 multi-class model
├── spam_model_v2.pkl          # Trained pipeline (preprocessor + TF-IDF + SVM)
├── model_v2_report.md         # Test set metrics and confusion matrix
├── requirements.txt           # Python dependencies
├── spam-classifier-frontend/
│   └── index.html             # Single-page frontend (dark/light mode)
└── email_dataset_v2.csv       # Combined training dataset (local only, not on GitHub)
```

---

## Run Locally

```bash
# Clone
git clone https://github.com/korie-cyber/End-to-End-Spam-Classier-.git
cd End-to-End-Spam-Classier-

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload

# Open http://localhost:8000
```

---

## Retrain the Model

If you have the 5 source CSVs in the project root:

```bash
# 1. Combine and clean datasets (produces email_dataset_v2.csv)
#    See train_model_v2.py for the full pipeline

# 2. Train
python train_model_v2.py

# This outputs spam_model_v2.pkl and model_v2_report.md
```

---

## Tech Stack

- **ML**: scikit-learn (LinearSVC + CalibratedClassifierCV, TF-IDF)
- **Backend**: FastAPI, uvicorn
- **Frontend**: Vanilla HTML/CSS/JS, Tabler Icons
- **Hosting**: Render (free tier, auto-deploys from GitHub)

---

## License

Open source. Built by [korie-cyber](https://github.com/korie-cyber).
