"""Word + character TF-IDF improvement experiment for Group Number 11.

Uses the fixed stratified 8,000/2,000 split (random_state=42).

Run:
    python run_word_char_improvement.py --train_csv /path/to/train.csv
"""
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from run_person_c_evaluation import load_fixed_split

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--out", default="word_char_improvement_results.csv")
    args = parser.parse_args()

    tr, va = load_fixed_split(args.train_csv)
    X_train, y_train = tr.text, tr.label.to_numpy()
    X_val, y_val = va.text, va.label.to_numpy()

    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
        )),
    ])

    X_train_f = features.fit_transform(X_train)
    X_val_f = features.transform(X_val)

    rows = []
    for C in [0.1, 0.25, 0.5, 1.0]:
        model = LinearSVC(C=C, random_state=SEED)
        start = time.perf_counter()
        model.fit(X_train_f, y_train)
        fit_time = time.perf_counter() - start

        rows.append({
            "model": "Word + Character TF-IDF + Linear SVM",
            "C": C,
            "train_auroc": roc_auc_score(
                y_train, model.decision_function(X_train_f)
            ),
            "validation_auroc": roc_auc_score(
                y_val, model.decision_function(X_val_f)
            ),
            "fit_time_seconds": fit_time,
        })

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    print(result.sort_values("validation_auroc", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
