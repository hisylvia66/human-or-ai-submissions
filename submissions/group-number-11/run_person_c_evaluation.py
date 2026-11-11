"""Q2/Q4: fixed-split learning curves and validation error analysis.

Run from any directory with --train_csv /path/to/competition/train.csv.
No competition text is written to the results folder. --review_text writes a
local, ignored file for qualitative inspection; do not commit that file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aml-person-c-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

TEAM = Path(__file__).resolve().parent
SEED = 42
SIZES = (500, 1000, 2000, 5000, 8000)


def make_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                    max_features=100000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                    min_df=2, max_features=150000, sublinear_tf=True)),
        ])),
        ("svm", LinearSVC(C=0.5, random_state=SEED)),
    ])


def load_fixed_split(path):
    data = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    if not {"id", "text", "label"}.issubset(data.columns):
        raise ValueError("train.csv must contain id, text, label")
    if data.id.isna().any() or data.id.duplicated().any():
        raise ValueError("Training IDs must be unique and nonmissing")
    if data.text.isna().any() or not data.label.isin([0, 1]).all():
        raise ValueError("Expected nonmissing text and binary labels")
    ids = [pd.read_csv(TEAM / name)["id"] for name in
           ("train_ids.csv", "validation_ids.csv")]
    for values, expected in zip(ids, (8000, 2000)):
        if len(values) != expected or values.isna().any() or values.duplicated().any():
            raise ValueError("Invalid committed split IDs")
    if (set(ids[0]) & set(ids[1])) or ((set(ids[0]) | set(ids[1])) != set(data.id)):
        raise ValueError("Committed split must partition the provided dataset exactly")
    data = data.set_index("id")
    tr, va = [data.loc[values].reset_index() for values in ids]
    if tr.label.value_counts().to_dict() != {0: 4000, 1: 4000}:
        raise ValueError("Expected balanced fixed 8,000-row training set")
    if va.label.value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Expected balanced fixed 2,000-row validation set")
    return tr, va


def nested_order(labels):
    rng = np.random.default_rng(SEED)
    # Interleave independently shuffled class pools: every requested prefix is balanced.
    pools = [rng.permutation(np.flatnonzero(np.asarray(labels) == label)) for label in (0, 1)]
    return np.column_stack(pools).ravel()


def auc(y, scores):
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=TEAM / "person_c_results")
    parser.add_argument("--review_text", type=Path,
                        help="Optional local-only CSV containing selected error texts")
    args = parser.parse_args()
    tr, va = load_fixed_split(args.train_csv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    order = nested_order(tr.label)
    rows, membership = [], []
    for size in SIZES:
        # Preserve the team's original row order at 8k for baseline reproduction.
        subset = tr if size == 8000 else tr.iloc[order[:size]]
        model = make_model()
        start = time.perf_counter()
        model.fit(subset.text, subset.label)
        fit_seconds = time.perf_counter() - start
        train_scores = model.decision_function(subset.text)
        val_scores = model.decision_function(va.text)
        row = dict(n_train=size, n_human=size // 2, n_ai=size // 2,
                   train_auroc=auc(subset.label, train_scores),
                   validation_auroc=auc(va.label, val_scores),
                   fit_seconds=fit_seconds)
        rows.append(row)
        membership.extend({"n_train": size, "id": i} for i in subset.id)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out / "learning_curve_results.csv", index=False)
    pd.DataFrame(membership).to_csv(out / "learning_curve_subset_ids.csv", index=False)
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for col, label in (("train_auroc", "Training"), ("validation_auroc", "Validation")):
        ax.plot(curve.n_train, curve[col], "o-", label=label)
    ax.set(xlabel="Number of labeled training examples", ylabel="AUROC",
           title="Fixed word + character TF-IDF / Linear SVM (C=0.5)", ylim=(0.5, 1.01))
    ax.set_xticks(SIZES)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=200)
    fig.savefig(out / "learning_curve.pdf")
    plt.close(fig)

    # Every fold refits TF-IDF as well as SVM; validation never enters calibration.
    print("Fitting probability calibration using 3 training-only folds...", flush=True)
    start = time.perf_counter()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_model(), tr.text, tr.label, cv=cv,
                           method="decision_function", n_jobs=1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
    calibrator.fit(oof.reshape(-1, 1), tr.label)
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration must preserve the selected SVM's ranking")
    probabilities = calibrator.predict_proba(val_scores.reshape(-1, 1))[:, 1]
    calibration_seconds = time.perf_counter() - start
    predictions = (val_scores > 0).astype(int)
    lengths = va.text.str.split().str.len()
    # Training-only tertiles define boundaries; ties stay in the same length group.
    boundaries = np.quantile(tr.text.str.split().str.len(), [1 / 3, 2 / 3])
    groups = pd.cut(lengths, [-np.inf, *boundaries, np.inf],
                    labels=["short", "medium", "long"], right=True)
    scores = pd.DataFrame({"id": va.id, "label": va.label,
                           "word_count": lengths, "length_group": groups,
                           "decision_score": val_scores, "p_ai": probabilities,
                           "predicted_label": predictions})
    scores["error_type"] = np.where((va.label == 0) & (predictions == 1), "FP",
                                    np.where((va.label == 1) & (predictions == 0), "FN", "correct"))
    scores.to_csv(out / "validation_predictions.csv", index=False)
    subgroup = []
    for group in ("short", "medium", "long"):
        frame = scores[scores.length_group == group]
        subgroup.append(dict(length_group=group, n=len(frame),
                             n_human=int((frame.label == 0).sum()), n_ai=int(frame.label.sum()),
                             min_words=int(frame.word_count.min()) if len(frame) else None,
                             max_words=int(frame.word_count.max()) if len(frame) else None,
                             validation_auroc=auc(frame.label, frame.decision_score),
                             false_positives=int((frame.error_type == "FP").sum()),
                             false_negatives=int((frame.error_type == "FN").sum())))
    pd.DataFrame(subgroup).to_csv(out / "length_group_results.csv", index=False)
    selected = pd.concat([scores[scores.error_type == kind].sort_values(
        "decision_score", ascending=(kind == "FN")).head(5) for kind in ("FP", "FN")])
    if len(selected) != 10:
        raise ValueError("Fewer than five errors in one class; inspect full predictions")
    selected.to_csv(out / "selected_error_predictions.csv", index=False)
    notes = json.loads((TEAM / "person_c_error_notes.json").read_text())
    annotated = selected.copy()
    for column in ("observed_pattern", "possible_reason"):
        annotated[column] = [notes.get(str(i), {}).get(column, "REQUIRES TEXT REVIEW")
                             for i in annotated.id]
    annotated.to_csv(out / "error_analysis.csv", index=False)
    if args.review_text:
        args.review_text.parent.mkdir(parents=True, exist_ok=True)
        selected.merge(va[["id", "text"]], on="id").to_csv(args.review_text, index=False)
    summary = dict(model="Word + Character TF-IDF + LinearSVC", C=0.5, seed=SEED,
                   n_train=len(tr), n_validation=len(va), train_auroc=rows[-1]["train_auroc"],
                   validation_auroc=rows[-1]["validation_auroc"],
                   train_validation_gap=rows[-1]["train_auroc"] - rows[-1]["validation_auroc"],
                   confusion_matrix_true_rows_predicted_columns=confusion_matrix(va.label, predictions).tolist(),
                   classification_threshold_margin=0.0,
                   classification_threshold_p_ai=float(calibrator.predict_proba([[0]])[0, 1]),
                   calibration="Logistic sigmoid fitted to 3-fold out-of-fold training margins; TF-IDF refit per fold",
                   calibration_slope=float(calibrator.coef_[0, 0]),
                   calibration_intercept=float(calibrator.intercept_[0]),
                   calibration_seconds=calibration_seconds,
                   length_measure="Whitespace-separated word count; training tertiles; right-closed groups",
                   length_boundaries=boundaries.tolist(),
                   error_selection="Five highest-margin FP and five lowest-margin FN; purposive, not representative",
                   train_csv_sha256=hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
                   split_sha256={name: hashlib.sha256((TEAM / name).read_bytes()).hexdigest()
                                 for name in ("train_ids.csv", "validation_ids.csv")},
                   versions={"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "matplotlib": matplotlib.__version__})
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
