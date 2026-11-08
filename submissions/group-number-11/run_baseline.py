"""Reproducible initial baseline for Cornell Tech Human or AI competition.
Run: python run_baseline.py --zip /path/to/competition.zip
Dependencies: pandas numpy scikit-learn scipy
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='cornell-tech-cs-5785-combined-2026-ai-vs-human-text.zip')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        train = pd.read_csv(z.open('train.csv'))
        test = pd.read_csv(z.open('test.csv'))
        sample = pd.read_csv(z.open('sample_submission.csv'))
    assert train.id.is_unique and train.label.notna().all() and train.text.notna().all()
    assert set(train.label.unique()) == {0, 1}
    train_index, val_index = train_test_split(
        np.arange(len(train)), test_size=2000, random_state=SEED,
        stratify=train['label'].to_numpy()
    )
    # Store IDs rather than transient row locations, so teammates can recreate identical splits.
    pd.DataFrame({'id': train.iloc[train_index].id}).to_csv(out/'train_ids.csv', index=False)
    pd.DataFrame({'id': train.iloc[val_index].id}).to_csv(out/'validation_ids.csv', index=False)
    x_train, y_train = train.iloc[train_index].text, train.iloc[train_index].label
    x_val, y_val = train.iloc[val_index].text, train.iloc[val_index].label
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)),
        ('logreg', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    start = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter()-start
    p_train = model.predict_proba(x_train)[:, 1]
    p_val = model.predict_proba(x_val)[:, 1]
    result = {
        'model': 'word TF-IDF (1,2)-gram + Logistic Regression',
        'random_state': SEED, 'train_n': len(train_index), 'validation_n':len(val_index),
        'train_class_counts': {str(k):int(v) for k,v in y_train.value_counts().items()},
        'validation_class_counts': {str(k):int(v) for k,v in y_val.value_counts().items()},
        'train_auroc': float(roc_auc_score(y_train, p_train)),
        'validation_auroc': float(roc_auc_score(y_val, p_val)),
        'fit_time_seconds': round(elapsed, 3),
        'n_features': int(len(model.named_steps['tfidf'].vocabulary_)),
        'params': {'ngram_range':[1,2], 'min_df':2, 'max_features':100000, 'sublinear_tf':True, 'C':1.0, 'max_iter':1000}
    }
    pd.DataFrame({'id': train.iloc[val_index].id.values, 'text': x_val.values,
                  'true_label': y_val.values, 'p_ai':p_val}).to_csv(out/'validation_predictions.csv', index=False)
    with open(out/'baseline_metrics.json','w') as f: json.dump(result, f, indent=2)
    # This is only a trial Kaggle submission using 8k labeled examples;
    # choose final model and retrain on 10k before final submission.
    pred = model.predict_proba(test.text)[:, 1]
    submission = sample[['id']].merge(pd.DataFrame({'id':test.id, 'label':pred}),on='id',suffixes=('_sample',''))
    submission = submission[['id','label']]
    assert len(submission)==len(test) and submission.label.between(0,1).all()
    submission.to_csv(out/'baseline_submission_8k.csv',index=False)
    print(json.dumps(result,indent=2))
    print('Output directory:',out.absolute())

if __name__ == '__main__': main()
