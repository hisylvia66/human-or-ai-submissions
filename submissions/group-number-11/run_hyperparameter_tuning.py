"""Hyperparameter tuning for Group Number 11.

Uses the fixed stratified 8,000/2,000 split with random_state=42.
Keeps the TF-IDF representation fixed and tunes only classifier C.

Run:
    python run_hyperparameter_tuning.py --train_csv /path/to/train.csv
"""
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--out", default="hyperparameter_tuning_results.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)

    idx = np.arange(len(train))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=2000,
        random_state=SEED,
        stratify=train["label"].to_numpy(),
    )

    y_train = train.iloc[train_idx]["label"].to_numpy()
    y_val = train.iloc[val_idx]["label"].to_numpy()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train.iloc[train_idx]["text"])
    X_val = vectorizer.transform(train.iloc[val_idx]["text"])

    rows = []

    for C in [0.1, 0.5, 1, 2, 5]:
        model = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        rows.append({
            "model": "Logistic Regression",
            "C": C,
            "train_auroc": roc_auc_score(
                y_train, model.predict_proba(X_train)[:, 1]
            ),
            "validation_auroc": roc_auc_score(
                y_val, model.predict_proba(X_val)[:, 1]
            ),
            "fit_time_seconds": fit_time,
        })

    for C in [0.01, 0.1, 0.5, 1, 2]:
        model = LinearSVC(C=C, random_state=SEED)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        rows.append({
            "model": "Linear SVM",
            "C": C,
            "train_auroc": roc_auc_score(
                y_train, model.decision_function(X_train)
            ),
            "validation_auroc": roc_auc_score(
                y_val, model.decision_function(X_val)
            ),
            "fit_time_seconds": fit_time,
        })

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    print(results.sort_values(
        ["model", "validation_auroc"],
        ascending=[True, False]
    ).to_string(index=False))

if __name__ == "__main__":
    main()
