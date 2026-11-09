"""Generate Group Number 11 Kaggle submission.

Current selected model:
- Word TF-IDF: 1-2 grams
- Character TF-IDF: 3-5 grams
- Linear SVM, C=0.5
- Train on all 10,000 labeled examples
- Convert decision scores to [0,1] with a sigmoid for Kaggle submission

Run:
python generate_submission.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_submission data/sample_submission.csv \
  --out submission.csv
"""

import argparse
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--sample_submission", default="data/sample_submission.csv")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    sample = pd.read_csv(args.sample_submission)

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

    X_train = features.fit_transform(train["text"])
    X_test = features.transform(test["text"])

    model = LinearSVC(C=0.5, random_state=SEED)
    model.fit(X_train, train["label"])

    raw_scores = model.decision_function(X_test)
    probabilities = expit(raw_scores)

    prediction_columns = [c for c in sample.columns if c.lower() != "id"]
    prediction_column = prediction_columns[0]

    sample[prediction_column] = probabilities
    sample.to_csv(args.out, index=False)

    print(f"Saved {args.out} with {len(sample)} rows.")

if __name__ == "__main__":
    main()
