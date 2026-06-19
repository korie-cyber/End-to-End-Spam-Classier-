"""
Train multi-class email threat classifier (ham / spam / phishing).

Dataset: email_dataset_v2.csv (combined from 5 sources)
Pipeline: TextPreprocessor -> TF-IDF -> Classifier
Outputs:  spam_model_v2.pkl, model_v2_report.md
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from preprocessing import TextPreprocessor

np.random.seed(42)

DATASET = "email_dataset_v2.csv"
MODEL_OUT = "spam_model_v2.pkl"
REPORT_OUT = "model_v2_report.md"

LABELS = ["ham", "spam", "phishing"]
LABEL_MAP = {"ham": 0, "spam": 1, "phishing": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


# ═══════════════════════════════════════════════════════
# 1. LOAD + BALANCE
# ═══════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATASET, encoding="utf-8")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABELS)].copy()

    # Cap ham at 2x phishing count
    phish_n = (df["label"] == "phishing").sum()
    ham_cap = int(phish_n * 2.0)

    ham = df[df["label"] == "ham"]
    if len(ham) > ham_cap:
        ham = ham.sample(n=ham_cap, random_state=42)

    df = pd.concat([
        ham,
        df[df["label"] == "spam"],
        df[df["label"] == "phishing"],
    ], ignore_index=True)

    X = df["text"]
    y = df["label"].map(LABEL_MAP)

    print(f"Dataset loaded: {len(df):,} rows")
    print("Class balance:")
    for lbl in LABELS:
        n = (df["label"] == lbl).sum()
        print(f"  {lbl:10s}  {n:>6,}  ({n/len(df)*100:.1f}%)")
    print()

    return X, y


# ═══════════════════════════════════════════════════════
# 2. SPLIT: 80/10/10 stratified
# ═══════════════════════════════════════════════════════

def split_data(X, y):
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Second split: 50/50 of the 20% -> 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  "
          f"Test: {len(X_test):,}")
    print()
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════
# 3. BUILD PIPELINES
# ═══════════════════════════════════════════════════════

def build_svc_pipeline():
    return Pipeline([
        ("preprocessor", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
            ),
            cv=5,
            method="sigmoid",
        )),
    ])


def build_lr_pipeline():
    return Pipeline([
        ("preprocessor", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])


# ═══════════════════════════════════════════════════════
# 4. EVALUATE
# ═══════════════════════════════════════════════════════

def evaluate(pipe, X, y, split_name):
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"=== {split_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print()
    print(classification_report(
        y, y_pred,
        target_names=LABELS,
        digits=4,
    ))

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"{'':>12s}  {'ham':>6s}  {'spam':>6s}  {'phish':>6s}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:>10s}  {cm[i,0]:>6d}  {cm[i,1]:>6d}  {cm[i,2]:>6d}")
    print()

    return acc, classification_report(
        y, y_pred, target_names=LABELS, digits=4, output_dict=True
    ), cm


def test_probabilities(pipe, X, n=3):
    """Show a few probability outputs to confirm calibration works."""
    probs = pipe.predict_proba(X.iloc[:n])
    print("Sample probability outputs:")
    for i in range(min(n, len(probs))):
        p = probs[i]
        pred = LABELS[np.argmax(p)]
        print(f"  [{pred:>8s}]  ham={p[0]:.3f}  spam={p[1]:.3f}  "
              f"phishing={p[2]:.3f}")
    print()


# ═══════════════════════════════════════════════════════
# 5. GENERATE REPORT
# ═══════════════════════════════════════════════════════

def write_report(model_name, acc, report_dict, cm, train_size, val_size,
                 test_size, class_balance):
    lines = [
        "# Model v2 — Multi-class Email Threat Classifier",
        "",
        "## Overview",
        "",
        f"- **Model**: {model_name}",
        f"- **Pipeline**: TextPreprocessor -> TF-IDF (30k features, "
        f"1-2 ngrams) -> {model_name}",
        f"- **Classes**: ham, spam, phishing",
        f"- **Split**: train={train_size:,} / val={val_size:,} / "
        f"test={test_size:,} (80/10/10, stratified)",
        "",
        "## Training Class Balance",
        "",
        "| Class | Count | % |",
        "|---|---|---|",
    ]

    total = sum(class_balance.values())
    for lbl in LABELS:
        n = class_balance[lbl]
        lines.append(f"| {lbl} | {n:,} | {n/total*100:.1f}% |")

    lines += [
        "",
        "## Test Set Results",
        "",
        f"**Overall accuracy: {acc:.4f}**",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]

    for lbl in LABELS:
        r = report_dict[lbl]
        lines.append(
            f"| {lbl} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1-score']:.4f} | {int(r['support'])} |"
        )

    lines += [
        "",
        "## Confusion Matrix",
        "",
        "Rows = actual, columns = predicted.",
        "",
        "| | ham | spam | phishing |",
        "|---|---|---|---|",
    ]

    for i, lbl in enumerate(LABELS):
        lines.append(f"| **{lbl}** | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} |")

    lines += [
        "",
        "## Phishing Focus",
        "",
    ]
    pr = report_dict["phishing"]
    lines.append(
        f"- **Precision**: {pr['precision']:.4f} "
        f"(of emails flagged phishing, {pr['precision']*100:.1f}% really are)")
    lines.append(
        f"- **Recall**: {pr['recall']:.4f} "
        f"(of real phishing emails, {pr['recall']*100:.1f}% are caught)")
    lines.append(
        f"- **F1**: {pr['f1-score']:.4f}")
    lines.append("")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved: {REPORT_OUT}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    class_balance = {}
    for lbl, code in LABEL_MAP.items():
        class_balance[lbl] = int((y_train == code).sum())

    # ── Train both models ─────────────────────────────
    print("Training LinearSVC + CalibratedClassifierCV ...")
    svc_pipe = build_svc_pipeline()
    svc_pipe.fit(X_train, y_train)
    print("Done.\n")

    print("Training LogisticRegression ...")
    lr_pipe = build_lr_pipeline()
    lr_pipe.fit(X_train, y_train)
    print("Done.\n")

    # ── Evaluate on VALIDATION set ────────────────────
    print("=" * 62)
    print("VALIDATION SET COMPARISON")
    print("=" * 62)
    print()

    print("[LinearSVC + Calibration]")
    svc_acc, svc_report, _ = evaluate(svc_pipe, X_val, y_val, "Validation")
    test_probabilities(svc_pipe, X_val)

    print("[LogisticRegression]")
    lr_acc, lr_report, _ = evaluate(lr_pipe, X_val, y_val, "Validation")
    test_probabilities(lr_pipe, X_val)

    # ── Pick winner ───────────────────────────────────
    svc_phish_f1 = svc_report["phishing"]["f1-score"]
    lr_phish_f1 = lr_report["phishing"]["f1-score"]
    svc_macro_f1 = svc_report["macro avg"]["f1-score"]
    lr_macro_f1 = lr_report["macro avg"]["f1-score"]

    print("=" * 62)
    print("HEAD-TO-HEAD (validation set)")
    print("=" * 62)
    print(f"{'Metric':<22s}  {'SVC+Cal':>8s}  {'LogReg':>8s}")
    print(f"{'Accuracy':<22s}  {svc_acc:>8.4f}  {lr_acc:>8.4f}")
    print(f"{'Macro F1':<22s}  {svc_macro_f1:>8.4f}  {lr_macro_f1:>8.4f}")
    print(f"{'Phishing F1':<22s}  {svc_phish_f1:>8.4f}  {lr_phish_f1:>8.4f}")
    print()

    # Prefer whichever has better phishing F1; break ties with macro F1
    if svc_phish_f1 >= lr_phish_f1:
        winner_pipe = svc_pipe
        winner_name = "LinearSVC + CalibratedClassifierCV"
        print(f"Winner: {winner_name}")
    else:
        winner_pipe = lr_pipe
        winner_name = "LogisticRegression (multinomial)"
        print(f"Winner: {winner_name}")
    print()

    # ── Final evaluation on TEST set ──────────────────
    print("=" * 62)
    print("FINAL TEST SET EVALUATION")
    print("=" * 62)
    print()
    test_acc, test_report, test_cm = evaluate(
        winner_pipe, X_test, y_test, "Test"
    )
    test_probabilities(winner_pipe, X_test)

    # ── Save model + report ───────────────────────────
    joblib.dump(winner_pipe, MODEL_OUT)
    print(f"Model saved: {MODEL_OUT}")

    write_report(
        winner_name, test_acc, test_report, test_cm,
        len(X_train), len(X_val), len(X_test), class_balance
    )

    # ── Example predictions ───────────────────────────
    examples = [
        "Congratulations! You've won a $1,000 gift card. Click "
        "http://claim-now.win to collect.",

        "Hi Mom, what time should I come over for dinner tonight?",

        "Your PayPal account has been limited. Verify your identity "
        "immediately at paypal-secure-verify.com/login or your account "
        "will be permanently suspended.",

        "Meeting is scheduled for tomorrow at 10 AM. Please bring "
        "your reports.",

        "URGENT: We detected unauthorized login to your bank account. "
        "Click here to verify: www.secure-banking-verify.com",

        "Please review the attached invoice and confirm receipt.",
    ]

    print()
    print("=" * 62)
    print("EXAMPLE PREDICTIONS")
    print("=" * 62)
    probs = winner_pipe.predict_proba(examples)
    preds = winner_pipe.predict(examples)

    for text, pred, prob in zip(examples, preds, probs):
        label = LABELS[pred]
        conf = prob[pred] * 100
        print(f"\n  [{label:>8s} {conf:5.1f}%]  {text[:75]}")
        print(f"  {'':>12s}  ham={prob[0]:.3f}  spam={prob[1]:.3f}  "
              f"phishing={prob[2]:.3f}")


if __name__ == "__main__":
    main()
