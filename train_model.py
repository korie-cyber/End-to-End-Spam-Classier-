"""
Retrain the spam classifier with improved feature engineering.

Key improvements over the original notebook:
- Smart token replacement (URLs, money, ALL-CAPS, repeated punctuation)
  instead of stripping those signals entirely
- class_weight='balanced' so the minority spam class is not drowned out
  by the 87:13 ham:spam ratio
- sublinear_tf=True in TF-IDF to dampen the effect of very frequent terms
- Trigrams (1,3) instead of bigrams (1,2) for richer phrase patterns
- Larger vocabulary cap (20 000 vs 10 000)
- Full sklearn Pipeline so training and inference preprocessing can never drift
"""

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib

from preprocessing import TextPreprocessor


SPAM_CSV = "spam.csv"
MODEL_OUT = "spam_model.pkl"


def load_data(path: str) -> tuple:
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna()
    return df["message"], df["label"]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", TextPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    min_df=1,
                    max_df=0.90,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)

    print("\n=== Test Set Results ===")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion matrix:")
    print(f"  Correct ham  (TN): {tn:4d}   Legit flagged as spam (FP): {fp:4d}")
    print(f"  Spam missed  (FN): {fn:4d}   Spam caught           (TP): {tp:4d}")
    print(f"\n  Spam recall (caught / total spam): {tp / (tp + fn):.2%}")
    print(f"  Spam precision   (of flagged, real spam): {tp / (tp + fp):.2%}")


def test_examples(pipeline: Pipeline) -> None:
    examples = [
        # (text, expected_label)
        ("Congratulations! You've won a $1,000 gift card. Click http://claim-now.win to collect.", 1),
        ("Hi Mom, what time should I come over for dinner tonight?", 0),
        ("URGENT: Your account will be closed. Verify at www.secure-bank.net now!", 1),
        ("Meeting is scheduled for tomorrow at 10 AM. Please bring your reports.", 0),
        ("FREE Entry! Text WIN to 87077 to claim your prize worth £500!", 1),
        ("Hey, are we still on for lunch on Friday?", 0),
        ("You have been selected for a £2000 award. Call 08712300440 to claim.", 1),
        ("The document you requested is attached. Let me know if you have questions.", 0),
        ("WIN a brand new iPhone! Reply YES to 80488 to enter our FREE prize draw!", 1),
        ("Please review the attached invoice and confirm receipt.", 0),
        ("Dear customer, your loan application has been approved. Call us at 1-800-555-0199.", 1),
        ("Can you send me the slides from today's presentation? Thanks!", 0),
    ]

    texts = [e[0] for e in examples]
    preds = pipeline.predict(texts)

    correct = 0
    print("\n=== Example Predictions ===")
    for (text, expected), pred in zip(examples, preds):
        mark = "OK" if pred == expected else "XX"
        label = "SPAM" if pred == 1 else "HAM "
        if pred == expected:
            correct += 1
        print(f"  {mark} [{('HAM ','SPAM')[expected]}->{label}] {text[:72]}")
    print(f"\n  {correct}/{len(examples)} examples correct")


def main() -> None:
    print("Loading data ...")
    X, y = load_data(SPAM_CSV)
    print(f"  {len(y)} samples | spam: {y.sum()} ({y.mean():.1%}) | ham: {(y==0).sum()} ({(y==0).mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    print("\nCross-validating on training set (5-fold stratified) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  F1       : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}")
    print(f"  Accuracy : {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")

    print("\nTraining final model on full training set ...")
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_test, y_test)
    test_examples(pipeline)

    joblib.dump(pipeline, MODEL_OUT)
    print(f"\nModel saved -> {MODEL_OUT}")
    print("Deploy: push spam_model.pkl and preprocessing.py to your server.")


if __name__ == "__main__":
    main()
