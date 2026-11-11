"""Shared model definitions for controlled refinement, final evaluation and submission."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

SEED = 42
CONFIGS = {
    'baseline': 'Word + character TF-IDF / SVM C=0.5',
    'char_weight2': 'Character block weighted 2x / SVM C=0.5',
    'char_only': 'Character TF-IDF only / SVM C=0.5',
    'logreg': 'Word + character TF-IDF / Logistic Regression C=5',
    'nbsvm': 'Binary document log-count-ratio weighted TF-IDF / SVM C=0.5',
    'selftrain2': 'Two-round self-training, confidence 0.90, SVM C=0.5',
}


def features(config='baseline'):
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                          max_features=150000, sublinear_tf=True)
    if config == 'char_only':
        return char
    return FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                max_features=100000, sublinear_tf=True)),
        ('char', char),
    ], transformer_weights={'word': 1., 'char': 2.} if config == 'char_weight2' else None)


def sigmoid_calibrator(scores, y):
    model = LogisticRegression(C=1e6, solver='lbfgs', random_state=SEED)
    model.fit(np.asarray(scores).reshape(-1, 1), y)
    if model.coef_[0, 0] <= 0:
        raise ValueError('Expected increasing margin-to-probability calibration')
    return model


def oof_scores(config, texts, y, unlabeled=None):
    texts, y = np.asarray(texts), np.asarray(y)
    out = np.zeros(len(y))
    for train, held in StratifiedKFold(3, shuffle=True, random_state=SEED).split(texts, y):
        # Full feature extraction is refit in each fold; no held-out texts are fitted.
        model = TextModel(config).fit(texts[train], y[train], unlabeled)
        out[held] = model.decision_function(texts[held])
    return out


class TextModel:
    def __init__(self, config='baseline'):
        if config not in CONFIGS:
            raise ValueError(f'Unknown config: {config}')
        self.config = config

    def fit(self, texts, y, unlabeled=None):
        texts, y = np.asarray(texts), np.asarray(y)
        self.vectorizer = features(self.config)
        X = self.vectorizer.fit_transform(texts).tocsr()
        self.ratio = None
        self.rounds = []
        if self.config == 'nbsvm':
            binary = (X > 0).astype(float)
            counts = [1 + np.asarray(binary[y == label].sum(axis=0)).ravel() for label in (0, 1)]
            self.ratio = np.log((counts[1] / counts[1].sum()) / (counts[0] / counts[0].sum()))
            X = X.multiply(self.ratio).tocsr()
        if self.config == 'logreg':
            self.clf = LogisticRegression(C=5, solver='liblinear', max_iter=1000, random_state=SEED)
        else:
            self.clf = LinearSVC(C=0.5, random_state=SEED)
        self.clf.fit(X, y)
        if self.config == 'selftrain2':
            if unlabeled is None or len(unlabeled) == 0:
                raise ValueError('selftrain2 needs unlabeled text')
            # Freeze the training-only, fully out-of-fold sigmoid across rounds.
            # This avoids including pseudo-labels generated using held-fold targets
            # in calibration folds; it differs from Person B's exploratory procedure.
            calibrator = sigmoid_calibrator(oof_scores('baseline', texts, y), y)
            Xu = self.vectorizer.transform(unlabeled)
            for round_number in (1, 2):
                prob = calibrator.predict_proba(self.clf.decision_function(Xu).reshape(-1, 1))[:, 1]
                use = (prob >= 0.9) | (prob <= 0.1)
                pseudo = (prob[use] >= 0.5).astype(int)
                self.clf = LinearSVC(C=0.5, random_state=SEED).fit(
                    sp.vstack([X, Xu[use]], format='csr'), np.r_[y, pseudo])
                self.rounds.append({'round': round_number, 'n_pseudo': int(use.sum()),
                                    'n_pseudo_ai': int(pseudo.sum())})
        return self

    def decision_function(self, texts):
        X = self.vectorizer.transform(texts)
        if self.ratio is not None:
            X = X.multiply(self.ratio).tocsr()
        return self.clf.decision_function(X)
