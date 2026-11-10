"""Q3 - Does unlabeled.csv help? (Person B, Group Number 11)

Model: the team's current best, word (1-2) + char (3-5) TF-IDF + LinearSVC(C=0.5), on the fixed
8,000 / 2,000 split (train_ids.csv / validation_ids.csv).

Experiments, all scored on the same 2,000 validation rows:
  E0  supervised only.
  E1  unsupervised feature learning: TF-IDF vocabulary + IDF fit on train + unlabeled text;
      the classifier is still trained on labeled rows only.
  E2  self-training, 1 round: score the 3,000 unlabeled texts, pseudo-label the most confident
      fraction q (class-balanced: top q/2 by SVM margin -> AI, bottom q/2 -> human), retrain.
      q in {10%, 25%, 50%, 75%, 100%}.
  E3  iterative self-training, 3 rounds at the best q from E2.
Diagnostics:
  D1  pseudo-label noise: hold back 3,000 *labeled* train rows, pseudo-label them with a model
      trained on the other 5,000, and measure the error rate at each q (the real unlabeled set
      has no labels, so this is the only way to measure it).
  D2  domain shift: cross-validated AUROC of a classifier that separates train vs unlabeled,
      train vs test, and unlabeled vs test text (0.5 means indistinguishable). Uses text only, no
      test labels.

Leakage protection: validation rows are never used to fit the vectorizer, train the
pseudo-labeler, or choose pseudo-labels; they are only scored at the end.

Run:  python run_unlabeled_experiment.py            (downloads data if needed)
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_utils import SEED, load_split, load_test, load_unlabeled

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
C = 0.5


def features():  # identical to run_word_char_improvement.py
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=150000, sublinear_tf=True)),
    ])


def svm():
    return LinearSVC(C=C, random_state=SEED)


def select_pseudo(scores: np.ndarray, frac: float):
    """Class-balanced pseudo-labels: the k highest margins -> AI (1), k lowest -> human (0)."""
    k = int(round(frac * len(scores) / 2))
    order = np.argsort(scores)
    idx = np.r_[order[-k:], order[:k]]
    lab = np.r_[np.ones(k, int), np.zeros(k, int)]
    return idx, lab


def self_train(Xtr, ytr, Xun, Xva, yva, frac, rounds=1):
    clf = svm().fit(Xtr, ytr)
    hist = []
    for r in range(rounds):
        idx, lab = select_pseudo(clf.decision_function(Xun), frac)  # re-score all every round
        clf = svm().fit(sp.vstack([Xtr, Xun[idx]]), np.r_[ytr, lab])
        hist.append({"round": r + 1, "n_pseudo": len(idx),
                     "val_auroc": roc_auc_score(yva, clf.decision_function(Xva))})
    return clf, hist


def domain_auroc(a, b):
    X = list(a) + list(b)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    m = Pipeline([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                                        sublinear_tf=True)),
                  ("c", LogisticRegression(max_iter=1000))])
    p = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="unlabeled_experiment_results.csv")
    args = ap.parse_args()

    tr, va = load_split(args.data_dir)
    un = load_unlabeled(args.data_dir)
    test = load_test(args.data_dir)
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    rows = []

    def log(**r):
        rows.append(r)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()))

    # ---- features fit on labeled train text only (shared by E0, E2, E3)
    t0 = time.perf_counter()
    feat = features().fit(tr.text)
    Xtr, Xva, Xun = feat.transform(tr.text), feat.transform(va.text), feat.transform(un.text)
    feat_s = time.perf_counter() - t0
    print(f"[features] {Xtr.shape[1]} dims, {feat_s:.1f}s")

    print("\nE0 supervised only")
    t0 = time.perf_counter()
    base = svm().fit(Xtr, ytr)
    base_val = roc_auc_score(yva, base.decision_function(Xva))
    log(experiment="E0", method="supervised only (8k labeled)", n_pseudo=0,
        train_auroc=roc_auc_score(ytr, base.decision_function(Xtr)), val_auroc=base_val,
        runtime_s=feat_s + time.perf_counter() - t0)
    m_va, m_un = np.abs(base.decision_function(Xva)), np.abs(base.decision_function(Xun))
    print(f"  mean |margin|: val={m_va.mean():.3f}  unlabeled={m_un.mean():.3f}  "
          f"(lower on unlabeled = model is less sure there)")
    print(f"  predicted AI-rate: val={np.mean(base.decision_function(Xva) > 0):.3f}  "
          f"unlabeled={np.mean(base.decision_function(Xun) > 0):.3f}")

    print("\nE1 TF-IDF vocabulary/IDF fit on train + unlabeled text")
    t0 = time.perf_counter()
    f1 = features().fit(pd.concat([tr.text, un.text]))
    c1 = svm().fit(f1.transform(tr.text), ytr)
    log(experiment="E1", method="TF-IDF fit on train+unlabeled", n_pseudo=0,
        train_auroc=roc_auc_score(ytr, c1.decision_function(f1.transform(tr.text))),
        val_auroc=roc_auc_score(yva, c1.decision_function(f1.transform(va.text))),
        runtime_s=time.perf_counter() - t0)

    print("\nE2 self-training, 1 round")
    e2 = []
    for q in FRACTIONS:
        t0 = time.perf_counter()
        clf, h = self_train(Xtr, ytr, Xun, Xva, yva, q)
        e2.append((q, h[-1]["val_auroc"]))
        log(experiment="E2", method=f"self-training top {int(q*100)}%", n_pseudo=h[-1]["n_pseudo"],
            train_auroc=roc_auc_score(ytr, clf.decision_function(Xtr)), val_auroc=h[-1]["val_auroc"],
            runtime_s=feat_s + time.perf_counter() - t0)

    best_q = max(e2, key=lambda r: r[1])[0]
    print(f"\nE3 iterative self-training, 3 rounds, top {int(best_q*100)}%")
    t0 = time.perf_counter()
    clf, hist = self_train(Xtr, ytr, Xun, Xva, yva, best_q, rounds=3)
    for h in hist:
        log(experiment="E3", method=f"self-training top {int(best_q*100)}% round {h['round']}",
            n_pseudo=h["n_pseudo"],
            train_auroc=roc_auc_score(ytr, clf.decision_function(Xtr)) if h["round"] == 3 else np.nan,
            val_auroc=h["val_auroc"],
            runtime_s=feat_s + time.perf_counter() - t0 if h["round"] == 3 else np.nan)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    print("\nD1 pseudo-label error rate (simulated on held-back labeled rows)")
    a, b = train_test_split(np.arange(len(tr)), test_size=3000, stratify=ytr, random_state=SEED)
    sim = svm().fit(Xtr[a], ytr[a])
    s = sim.decision_function(Xtr[b])
    noise = []
    for q in FRACTIONS:
        idx, lab = select_pseudo(s, q)
        err = float(np.mean(lab != ytr[b][idx]))
        noise.append({"fraction": q, "n_selected": len(idx), "pseudo_label_error": err})
        print(f"  top {int(q*100):3d}%  n={len(idx):5d}  error={err:.3f}")
    pd.DataFrame(noise).to_csv("unlabeled_pseudo_label_noise.csv", index=False)

    print("\nD2 domain shift (5-fold CV AUROC of telling two text sets apart)")
    dom = [{"pair": "train vs unlabeled", "auroc": domain_auroc(tr.text, un.text)},
           {"pair": "train vs test", "auroc": domain_auroc(tr.text, test.text)},
           {"pair": "unlabeled vs test", "auroc": domain_auroc(un.text, test.text)}]
    for d in dom:
        print(f"  {d['pair']:<20} {d['auroc']:.3f}")
    pd.DataFrame(dom).to_csv("unlabeled_domain_shift.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([f"{int(q*100)}%" for q, _ in e2], [v for _, v in e2], marker="o",
            label="self-training (1 round)")
    ax.axhline(base_val, ls="--", c="gray", label=f"supervised only ({base_val:.4f})")
    ax.set_xlabel("fraction of unlabeled set pseudo-labeled")
    ax.set_ylabel("validation AUROC")
    ax.set_title("Q3: self-training vs supervised baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("unlabeled_self_training.png", dpi=200)
    print(f"\nwrote {args.out}, unlabeled_pseudo_label_noise.csv, unlabeled_domain_shift.csv, "
          "unlabeled_self_training.png")


if __name__ == "__main__":
    main()
