"""Q1 model comparison for Group Number 11.

Uses the required fixed 8,000/2,000 stratified split (random_state=42).
Run:
    python run_model_comparison.py --zip /path/to/competition.zip

Models:
1) Word TF-IDF + Logistic Regression
2) Word TF-IDF + Linear SVM
3) Word TF-IDF + Complement Naive Bayes
"""
import argparse
import time
import zipfile
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from run_person_c_evaluation import load_fixed_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42

def scores(model, X_train, y_train, X_val, y_val):
    if hasattr(model[-1], "predict_proba"):
        train_score = model.predict_proba(X_train)[:, 1]
        val_score = model.predict_proba(X_val)[:, 1]
    else:
        train_score = model.decision_function(X_train)
        val_score = model.decision_function(X_val)
    return (
        roc_auc_score(y_train, train_score),
        roc_auc_score(y_val, val_score),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--zip", default="cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip")
    parser.add_argument("--out", default="model_comparison_results.csv")
    args = parser.parse_args()

    if args.train_csv:
        tr, va = load_fixed_split(args.train_csv)
    else:
        with zipfile.ZipFile(args.zip) as z:
            tr, va = load_fixed_split(pd.read_csv(z.open("train.csv")))
    X_train, y_train = tr.text, tr.label
    X_val, y_val = va.text, va.label

    def tfidf():
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )

    models = {
        "Word TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "Word TF-IDF + Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LinearSVC(C=1.0, random_state=SEED)),
        ]),
        "Word TF-IDF + Complement Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf", ComplementNB(alpha=1.0)),
        ]),
    }

    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        train_auc, val_auc = scores(model, X_train, y_train, X_val, y_val)
        rows.append({
            "model": name,
            "train_auroc": train_auc,
            "validation_auroc": val_auc,
            "fit_time_seconds": elapsed,
        })

    result = pd.DataFrame(rows).sort_values("validation_auroc", ascending=False)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
