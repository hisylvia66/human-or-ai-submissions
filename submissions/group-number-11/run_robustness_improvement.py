"""Q5 - Open-ended improvement: robustness to adversarially edited AI text (Person B, Group 11).

Hypothesis
  The test set includes adversarially modified AI passages. Our best model (word + char TF-IDF
  + LinearSVC) matches exact tokens and character sequences, so cheap edits (Cyrillic look-alike
  letters, zero-width characters, typos) create unseen features and push AI text toward
  "human". (1) Unicode normalization before vectorizing should undo the character-level tricks;
  (2) training on typo-perturbed copies of BOTH classes should reduce sensitivity to typos
  without teaching the model that "typos mean AI".

Configurations (same split, same features, same LinearSVC C=0.5, seed 42):
  R0  word TF-IDF + LinearSVC                          (Q1 word-only model, reference)
  R1  word + char TF-IDF + LinearSVC                   (team's current best = BASELINE)
  R2  R1 + normalize_text()                            (improvement 1)
  R3  R2 + typo-augmented copies of all train texts    (improvement 2)

Evaluation: clean validation + three attacked copies in which ONLY the AI-written validation
passages are edited (an adversary tries to evade detection; human text is left alone):
  typo        ~4% of words get an adjacent-letter swap or a deleted letter
  homoglyph   ~15% of the letters a,e,o,p,c,y,x,i replaced by Cyrillic look-alikes
  zero_width  zero-width spaces inserted inside ~20% of words
Paired bootstrap 95% CIs vs R1 are reported for every condition.
Diagnostic: how many texts in train / unlabeled / test already contain non-ASCII letters,
Cyrillic characters or zero-width characters (are these attacks actually present?).

Run:  python run_robustness_improvement.py            (downloads data if needed)
"""
from __future__ import annotations

import argparse
import random
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from data_utils import SEED, load_split, load_test, load_unlabeled

C = 0.5

# ----------------------------------------------------------------------------- normalization
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "ο": "o", "α": "a", "ν": "v", "ı": "i",
})
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def normalize_text(s: str) -> str:
    """NFKC normalization, homoglyph folding, zero-width/soft-hyphen removal, whitespace collapse."""
    s = unicodedata.normalize("NFKC", s).translate(ZERO_WIDTH).translate(HOMOGLYPHS)
    return " ".join(s.split())


# ----------------------------------------------------------------------------- attacks
CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}


def typo(t, rng, rate=0.04):
    out = []
    for w in t.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:] if rng.random() < 0.5 else w[:i] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def homoglyph(t, rng, rate=0.15):
    return "".join(CYR[ch] if ch in CYR and rng.random() < rate else ch for ch in t)


def zero_width(t, rng, rate=0.20):
    return " ".join(w[:len(w) // 2] + "​" + w[len(w) // 2:] if len(w) > 3 and rng.random() < rate
                    else w for w in t.split(" "))


ATTACKS = {"typo": typo, "homoglyph": homoglyph, "zero_width": zero_width}


def attack_ai_only(texts, labels, fn, seed=SEED):
    rng = random.Random(seed)
    return [fn(t, rng) if y == 1 else t for t, y in zip(texts, labels)]


# ----------------------------------------------------------------------------- models
def word_feats():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)


def word_char_feats():  # identical to run_word_char_improvement.py
    return FeatureUnion([
        ("word", word_feats()),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=150000, sublinear_tf=True)),
    ])


CONFIGS = {
    "R0": ("word TF-IDF + LinearSVC (Q1 reference)", word_feats, False, False),
    "R1": ("word+char TF-IDF + LinearSVC (BASELINE)", word_char_feats, False, False),
    "R2": ("R1 + unicode normalization", word_char_feats, True, False),
    "R3": ("R2 + typo augmentation (both classes)", word_char_feats, True, True),
}


class Model:
    def __init__(self, feats, normalize, augment):
        self.feats, self.normalize, self.augment = feats(), normalize, augment

    def _prep(self, texts):
        return [normalize_text(t) for t in texts] if self.normalize else list(texts)

    def fit(self, texts, y):
        texts, y = list(texts), np.asarray(y)
        if self.augment:
            rng = random.Random(SEED + 1)
            texts = texts + [typo(t, rng) for t in texts]
            y = np.r_[y, y]
        X = self.feats.fit_transform(self._prep(texts))
        self.clf = LinearSVC(C=C, random_state=SEED).fit(X, y)
        return self

    def score(self, texts):
        return self.clf.decision_function(self.feats.transform(self._prep(texts)))


def paired_bootstrap(y, a, b, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(np.mean(d)), *np.percentile(d, [2.5, 97.5])


def char_census(name, texts, labels=None):
    s = pd.Series(list(texts))
    flags = pd.DataFrame({
        "non_ascii_letter": s.map(lambda t: any(ord(c) > 127 and c.isalpha() for c in t)),
        "cyrillic": s.map(lambda t: bool(re.search("[Ѐ-ӿ]", t))),
        "zero_width": s.map(lambda t: bool(re.search("[​‌‍⁠﻿­]", t))),
        "changed_by_normalize": s.map(lambda t: normalize_text(t) != " ".join(t.split())),
    })
    out = []
    groups = [("all", np.ones(len(s), bool))]
    if labels is not None:
        groups += [("human", np.asarray(labels) == 0), ("AI", np.asarray(labels) == 1)]
    for g, m in groups:
        out.append({"set": name, "group": g, "n": int(m.sum()),
                    **{k: int(flags[k][m].sum()) for k in flags}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--configs", default="R0,R1,R2,R3")
    ap.add_argument("--out", default="robustness_improvement_results.csv")
    args = ap.parse_args()

    tr, va = load_split(args.data_dir)
    un, te = load_unlabeled(args.data_dir), load_test(args.data_dir)
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()

    print("\nDiagnostic: suspicious characters already present in the data")
    census = (char_census("train(8k)", tr.text, ytr) + char_census("val", va.text, yva)
              + char_census("unlabeled", un.text) + char_census("test", te.text))
    census = pd.DataFrame(census)
    print(census.to_string(index=False))
    census.to_csv("robustness_char_census.csv", index=False)

    conditions = {"clean": list(va.text)}
    conditions.update({k: attack_ai_only(va.text, yva, fn) for k, fn in ATTACKS.items()})

    scores, rows = {}, []
    for cfg in args.configs.split(","):
        desc, feats, norm, aug = CONFIGS[cfg]
        print(f"\n{cfg}: {desc}")
        t0 = time.perf_counter()
        m = Model(feats, norm, aug).fit(tr.text, ytr)
        fit_s = time.perf_counter() - t0
        row = {"config": cfg, "description": desc, "fit_s": round(fit_s, 2),
               "train_auroc": roc_auc_score(ytr, m.score(tr.text))}
        scores[cfg] = {}
        for cond, texts in conditions.items():
            s = m.score(texts)
            scores[cfg][cond] = s
            row[f"val_{cond}"] = roc_auc_score(yva, s)
            if cfg != "R1" and "R1" in scores:
                d, lo, hi = paired_bootstrap(yva, scores["R1"][cond], s)
                row[f"delta_vs_R1_{cond}"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                               if k.startswith("val_") or k == "train_auroc") + f"  fit={fit_s:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + df.round(4).to_string(index=False))

    preds = pd.DataFrame({"id": va.id, "true_label": yva,
                          "length_words": va.text.str.split().str.len()})
    for cfg in scores:
        preds[f"score_{cfg}"] = scores[cfg]["clean"]
    preds.to_csv("robustness_validation_scores.csv", index=False)
    print(f"\nwrote {args.out}, robustness_char_census.csv, robustness_validation_scores.csv")


if __name__ == "__main__":
    main()
