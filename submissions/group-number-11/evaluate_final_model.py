"""Re-evaluate the selected final model on fixed IDs, including calibrated Q4 errors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix
from final_model import TextModel, oof_scores, sigmoid_calibrator, SEED
from run_person_c_evaluation import load_fixed_split

TEAM = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, default=TEAM / 'final_results')
    ap.add_argument('--review_text', type=Path, default=TEAM / 'data/final_error_review.csv')
    args = ap.parse_args()
    out = args.out_dir
    selection = json.loads((out / 'selected_model.json').read_text())
    config = selection['config']
    tr, va = load_fixed_split(args.data_dir / 'train.csv')
    un = pd.read_csv(args.data_dir / 'unlabeled.csv')
    norm = lambda s: s.str.lower().str.split().str.join(' ')
    un = un.loc[~norm(un.text).isin(set(norm(va.text)))]
    start = time.perf_counter()
    model = TextModel(config).fit(tr.text, tr.label, un.text)
    fit_s = time.perf_counter() - start
    train_scores, val_scores = model.decision_function(tr.text), model.decision_function(va.text)
    expected = pd.read_csv(out / 'refinement_validation_scores.csv')
    if va.id.tolist() != expected.id.tolist() or not np.allclose(val_scores, expected[config], atol=1e-10):
        raise ValueError('Selected-model reproduction mismatch')
    print('Refit matches selection; fitting training-only probability calibration.', flush=True)
    start = time.perf_counter()
    calibrator = sigmoid_calibrator(oof_scores(config, tr.text, tr.label, un.text), tr.label)
    calibration_s = time.perf_counter() - start
    p = calibrator.predict_proba(val_scores.reshape(-1, 1))[:, 1]
    bounds = np.quantile(tr.text.str.split().str.len(), [1/3, 2/3])
    counts = va.text.str.split().str.len()
    scores = pd.DataFrame(dict(id=va.id, label=va.label, word_count=counts,
                               length_group=pd.cut(counts, [-np.inf, *bounds, np.inf],
                                                   labels=['short', 'medium', 'long']),
                               decision_score=val_scores, p_ai=p,
                               predicted_label=(val_scores > 0).astype(int)))
    scores['error_type'] = np.where((scores.label == 0) & (scores.predicted_label == 1), 'FP',
                                   np.where((scores.label == 1) & (scores.predicted_label == 0), 'FN', 'correct'))
    scores.to_csv(out / 'validation_predictions.csv', index=False)
    groups = []
    for name in ['short','medium','long']:
        g = scores[scores.length_group == name]
        groups.append(dict(length_group=name, n=len(g), n_human=int((g.label == 0).sum()),
                           n_ai=int(g.label.sum()), min_words=int(g.word_count.min()), max_words=int(g.word_count.max()),
                           validation_auroc=roc_auc_score(g.label, g.decision_score),
                           false_positives=int((g.error_type == 'FP').sum()),
                           false_negatives=int((g.error_type == 'FN').sum())))
    pd.DataFrame(groups).to_csv(out / 'length_group_results.csv', index=False)
    errors = pd.concat([scores[scores.error_type == kind].sort_values(
        'decision_score', ascending=(kind == 'FN')).head(5) for kind in ('FP','FN')])
    if errors.error_type.value_counts().to_dict() != {'FP':5,'FN':5}:
        raise ValueError('Expected five errors of each type')
    notes_path = TEAM / 'final_error_notes.json'
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    for col in ('observed_pattern','possible_reason'):
        errors[col] = [notes.get(str(i),{}).get(col,'REQUIRES TEXT REVIEW') for i in errors.id]
    errors.to_csv(out / 'error_analysis.csv', index=False)
    args.review_text.parent.mkdir(parents=True, exist_ok=True)
    errors.merge(va[['id','text']], on='id').to_csv(args.review_text,index=False)
    summary = dict(config=config, seed=SEED, n_train=len(tr), n_validation=len(va),
                   train_auroc=roc_auc_score(tr.label, train_scores), validation_auroc=roc_auc_score(va.label, val_scores),
                   fit_seconds=fit_s, calibration_seconds=calibration_s,
                   classification_threshold_margin=0., classification_threshold_p_ai=float(calibrator.predict_proba([[0]])[0,1]),
                   calibration_slope=float(calibrator.coef_[0,0]), calibration_intercept=float(calibrator.intercept_[0]),
                   calibration='Sigmoid fitted on full-pipeline 3-fold out-of-fold training scores',
                   length_boundaries=bounds.tolist(),
                   confusion_matrix=confusion_matrix(va.label, scores.predicted_label).tolist(),
                   train_csv_sha256=hashlib.sha256((args.data_dir/'train.csv').read_bytes()).hexdigest(),
                   versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                 scipy=scipy.__version__,sklearn=sklearn.__version__))
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
