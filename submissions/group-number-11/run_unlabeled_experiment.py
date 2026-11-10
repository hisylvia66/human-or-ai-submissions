"""Q3 - Does unlabeled.csv help? (Person B, Group Number 11)

Model: the team's selected model, word (1-2) + char (3-5) TF-IDF + LinearSVC(C=0.5), on the
fixed 8,000 / 2,000 split (train_ids.csv / validation_ids.csv).

Method: confidence-based pseudo-labeling / self-training.
  1. Train the supervised model on the 8,000 training rows.
  2. Turn SVM margins into P(AI) with Platt scaling fit on out-of-fold training scores
     (LinearSVC has no probabilities of its own).
  3. Pseudo-label unlabeled rows with P(AI) >= t (AI) or P(AI) <= 1 - t (human).
  4. Retrain on labeled + pseudo-labeled rows and score the same 2,000 validation rows.

Experiments:
  E0  supervised only
  E1  TF-IDF vocabulary + IDF fit on train + unlabeled text (no pseudo-labels)
  E2  self-training, 1 round, t in {0.99, 0.95, 0.90, 0.80, 0.50 (= every row)}
  E3  iterative self-training, 3 rounds at the best t from E2
Diagnostics:
  D1  pseudo-label noise: hold back 3,000 *labeled* train rows, pseudo-label them with a model
      trained on the other 5,000, and measure the error rate at each t (the real unlabeled set
      has no labels, so this is an in-distribution estimate and probably optimistic).
  D2  domain shift: cross-validated AUROC of a classifier separating train vs unlabeled,
      train vs test and unlabeled vs test text (0.5 = indistinguishable). Text only; no test
      labels are involved.
  D3  how confident the supervised model is on validation vs unlabeled vs test text.

Leakage protection: validation rows are never used to fit the vectorizer, the SVM, the Platt
scaler or to choose pseudo-labels; they are only scored at the end.

Run:  python run_unlabeled_experiment.py            (finds / downloads the data itself)
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_utils import SEED, load_split, load_test, load_unlabeled, paired_bootstrap

THRESHOLDS = [0.99, 0.95, 0.90, 0.80, 0.50]  # 0.50 = pseudo-label every unlabeled row
C = 0.5


def features():  # identical to run_word_char_improvement.py / generate_submission.py
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=150000, sublinear_tf=True)),
    ])


def svm():
    return LinearSVC(C=C, random_state=SEED)


def fit_platt(X_lab, y_lab, X_extra=None, y_extra=None):
    """Platt scaler (margin -> P(AI)) from 5-fold out-of-fold margins on the labeled rows.
    Pseudo-labeled rows, if any, are added to every training fold but never used as targets."""
    oof = np.zeros(len(y_lab))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_lab, y_lab):
        Xa, ya = X_lab[a], y_lab[a]
        if X_extra is not None and X_extra.shape[0]:
            Xa, ya = sp.vstack([Xa, X_extra]), np.r_[ya, y_extra]
        oof[b] = svm().fit(Xa, ya).decision_function(X_lab[b])
    platt = LogisticRegression().fit(oof.reshape(-1, 1), y_lab)
    return lambda margins: platt.predict_proba(np.asarray(margins).reshape(-1, 1))[:, 1]


def select_pseudo(p_ai: np.ndarray, thr: float):
    """Rows whose confidence max(p, 1-p) >= thr, and their pseudo-labels."""
    idx = np.where(np.maximum(p_ai, 1 - p_ai) >= thr)[0]
    return idx, (p_ai[idx] >= 0.5).astype(int)


def self_train(Xtr, ytr, Xun, thr, rounds=1, on_round=None):
    """Returns the final classifier and per-round info. Every round re-scores all unlabeled rows."""
    clf, to_proba = svm().fit(Xtr, ytr), fit_platt(Xtr, ytr)
    info = []
    for r in range(rounds):
        idx, lab = select_pseudo(to_proba(clf.decision_function(Xun)), thr)
        clf = svm().fit(sp.vstack([Xtr, Xun[idx]]), np.r_[ytr, lab])
        info.append({"round": r + 1, "n_pseudo": len(idx),
                     "pseudo_ai_rate": float(lab.mean()) if len(lab) else float("nan")})
        if on_round:
            on_round(clf, info[-1])
        if r + 1 < rounds:
            to_proba = fit_platt(Xtr, ytr, Xun[idx], lab)
    return clf, info


def domain_auroc(a, b):
    X = list(a) + list(b)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    m = Pipeline([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                                        sublinear_tf=True)),
                  ("c", LogisticRegression(max_iter=1000))])
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="unlabeled_experiment_results.csv")
    args = ap.parse_args()

    tr, va = load_split(args.data_dir)
    un, test = load_unlabeled(args.data_dir), load_test(args.data_dir)
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    rows = []

    def log(**r):
        rows.append(r)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.items()))

    # ---- features fit on labeled training text only (shared by E0, E2, E3)
    t0 = time.perf_counter()
    feat = features().fit(tr.text)
    Xtr, Xva, Xun = feat.transform(tr.text), feat.transform(va.text), feat.transform(un.text)
    feat_s = time.perf_counter() - t0
    print(f"[features] {Xtr.shape[1]} dims, {feat_s:.1f}s")

    print("\nE0 supervised only")
    t0 = time.perf_counter()
    base = svm().fit(Xtr, ytr)
    s_base = base.decision_function(Xva)
    base_val = roc_auc_score(yva, s_base)
    log(experiment="E0", method="supervised only (8k labeled)", threshold=np.nan, n_pseudo=0,
        pseudo_ai_rate=np.nan, train_auroc=roc_auc_score(ytr, base.decision_function(Xtr)),
        val_auroc=base_val, runtime_s=feat_s + time.perf_counter() - t0)

    print("\nE1 TF-IDF vocabulary/IDF fit on train + unlabeled text")
    t0 = time.perf_counter()
    f1 = features().fit(pd.concat([tr.text, un.text]))
    c1 = svm().fit(f1.transform(tr.text), ytr)
    log(experiment="E1", method="TF-IDF fit on train+unlabeled", threshold=np.nan, n_pseudo=0,
        pseudo_ai_rate=np.nan,
        train_auroc=roc_auc_score(ytr, c1.decision_function(f1.transform(tr.text))),
        val_auroc=roc_auc_score(yva, c1.decision_function(f1.transform(va.text))),
        runtime_s=time.perf_counter() - t0)

    print("\nE2 self-training, 1 round, per confidence threshold")
    e2 = {}
    for thr in THRESHOLDS:
        t0 = time.perf_counter()
        clf, info = self_train(Xtr, ytr, Xun, thr)
        s = clf.decision_function(Xva)
        e2[thr] = s
        log(experiment="E2", method="self-training x1", threshold=thr, n_pseudo=info[-1]["n_pseudo"],
            pseudo_ai_rate=info[-1]["pseudo_ai_rate"],
            train_auroc=roc_auc_score(ytr, clf.decision_function(Xtr)),
            val_auroc=roc_auc_score(yva, s), runtime_s=feat_s + time.perf_counter() - t0)

    best_thr = max(e2, key=lambda t: roc_auc_score(yva, e2[t]))
    d, lo, hi = paired_bootstrap(yva, s_base, e2[best_thr])
    ci = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    print(f"  best threshold {best_thr}: val AUROC change vs E0 = {ci} (paired bootstrap 95% CI)")

    print(f"\nE3 iterative self-training, 3 rounds, threshold {best_thr}")
    t0 = time.perf_counter()
    self_train(Xtr, ytr, Xun, best_thr, rounds=3, on_round=lambda clf, i: log(
        experiment="E3", method=f"self-training round {i['round']}", threshold=best_thr,
        n_pseudo=i["n_pseudo"], pseudo_ai_rate=i["pseudo_ai_rate"],
        train_auroc=roc_auc_score(ytr, clf.decision_function(Xtr)),
        val_auroc=roc_auc_score(yva, clf.decision_function(Xva)),
        runtime_s=feat_s + time.perf_counter() - t0))
    res = pd.DataFrame(rows)
    res["best_vs_supervised_ci95"] = np.where(
        (res.experiment == "E2") & (res.threshold == best_thr), ci, "")
    res.to_csv(args.out, index=False)

    print("\nD1 pseudo-label error rate (simulated on held-back labeled rows)")
    a, b = train_test_split(np.arange(len(tr)), test_size=3000, stratify=ytr, random_state=SEED)
    p_sim = fit_platt(Xtr[a], ytr[a])(svm().fit(Xtr[a], ytr[a]).decision_function(Xtr[b]))
    noise = []
    for thr in THRESHOLDS:
        idx, lab = select_pseudo(p_sim, thr)
        err = float(np.mean(lab != ytr[b][idx])) if len(idx) else float("nan")
        noise.append({"threshold": thr, "n_selected": len(idx), "share_selected": len(idx) / len(b),
                      "pseudo_label_error": err})
        print(f"  t={thr:<5} selected={len(idx):5d} ({len(idx)/len(b):.0%})  error={err:.4f}")
    pd.DataFrame(noise).to_csv("unlabeled_pseudo_label_noise.csv", index=False)

    print("\nD2 domain shift (5-fold CV AUROC of telling two text sets apart; 0.5 = same)")
    dom = [{"pair": "train vs unlabeled", "auroc": domain_auroc(tr.text, un.text)},
           {"pair": "train vs test", "auroc": domain_auroc(tr.text, test.text)},
           {"pair": "unlabeled vs test", "auroc": domain_auroc(un.text, test.text)}]
    for r in dom:
        print(f"  {r['pair']:<20} {r['auroc']:.3f}")
    pd.DataFrame(dom).to_csv("unlabeled_domain_shift.csv", index=False)

    print("\nD3 supervised model's confidence on each set (calibrated on train)")
    to_proba = fit_platt(Xtr, ytr)
    conf = []
    for name, X in [("validation", Xva), ("unlabeled", Xun), ("test", feat.transform(test.text))]:
        p = to_proba(base.decision_function(X))
        c = np.maximum(p, 1 - p)
        conf.append({"set": name, "n": len(p), "predicted_ai_rate": float(np.mean(p >= 0.5)),
                     "mean_confidence": float(c.mean()),
                     **{f"share_conf>={t}": float(np.mean(c >= t)) for t in THRESHOLDS[:-1]}})
    conf = pd.DataFrame(conf)
    print(conf.round(3).to_string(index=False))
    conf.to_csv("unlabeled_confidence_by_set.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    e2_rows = res[res.experiment == "E2"]
    xs = [str(t) if t > 0.5 else "all" for t in e2_rows.threshold]
    ax.plot(xs, e2_rows.val_auroc, marker="o", label="self-training (1 round)")
    ax.axhline(base_val, ls="--", c="gray", label=f"supervised only ({base_val:.4f})")
    for x, v, n in zip(xs, e2_rows.val_auroc, e2_rows.n_pseudo):
        ax.annotate(f"+{n}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("pseudo-label confidence threshold (labels = rows added)")
    ax.set_ylabel("validation AUROC")
    ax.set_title("Q3: self-training vs supervised baseline")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("unlabeled_self_training.png", dpi=200)
    print(f"\nwrote {args.out}, unlabeled_pseudo_label_noise.csv, unlabeled_domain_shift.csv, "
          "unlabeled_confidence_by_set.csv, unlabeled_self_training.png")


if __name__ == "__main__":
    main()
