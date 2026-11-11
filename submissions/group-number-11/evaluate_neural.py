"""Compare the chosen local neural checkpoint with the submitted blend by ID.

No model fitting or test labels. Bootstrap intervals are descriptive and conditional
on checkpoint selection on this same holdout. Softmax outputs are uncalibrated.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss

TEAM = Path(__file__).resolve().parent
OUT = TEAM / 'neural_results'
summary = json.loads((OUT / 'validate_summary.json').read_text())
new = pd.read_csv(OUT / f"validate_epoch_{summary['best_epoch']}_scores.csv")
old = pd.read_csv(TEAM / 'audit_results/candidate_validation_predictions.csv')
expected = pd.read_csv(TEAM / 'validation_ids.csv').id
assert new.id.is_unique and old.id.is_unique
assert set(new.id) == set(old.id) == set(expected)
paired = old.merge(new, on='id', validate='one_to_one', suffixes=('_old', '_new'))
assert (paired.label_old == paired.label_new).all()
y = paired.label_new.to_numpy()
s = paired.decision_score_new.to_numpy()
baseline = paired.decision_score_old.to_numpy()
assert np.isfinite(s).all()
p = expit(s)
auc = roc_auc_score(y, s)
assert np.isclose(auc, summary['validation_auroc'])
rng = np.random.default_rng(42)
deltas = []
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) == 2:
        deltas.append(roc_auc_score(y[idx], s[idx]) - roc_auc_score(y[idx], baseline[idx]))
record = {
    'selected_epoch': summary['best_epoch'],
    'validation_auroc': auc,
    'submitted_blend_validation_auroc': roc_auc_score(y, baseline),
    'delta_vs_submitted_blend': auc - roc_auc_score(y, baseline),
    'paired_bootstrap_95ci_conditional_on_selection': np.quantile(deltas, [.025, .975]).tolist(),
    'confusion_matrix_at_logit_zero': confusion_matrix(y, s >= 0).tolist(),
    'uncalibrated_probability_brier_score': brier_score_loss(y, p),
    'public_score': None,
    'private_score': None,
    'limitations': 'Same validation set selects epoch; interval is descriptive, not independent confirmation. Softmax probabilities are not calibrated. Kaggle generalization remains unknown.'
}
rows = []
for group, frame in paired.groupby('length_group'):
    rows.append({'length_group': group, 'n': len(frame), 'n_ai': int(frame.label_new.sum()),
                 'validation_auroc': roc_auc_score(frame.label_new, frame.decision_score_new),
                 'submitted_blend_auroc': roc_auc_score(frame.label_old, frame.decision_score_old)})
pd.DataFrame(rows).to_csv(OUT / 'length_group_comparison.csv', index=False)
pred = pd.DataFrame({'id': paired.id, 'label': y, 'decision_score': s, 'p_ai_uncalibrated': p,
                     'word_count': paired.word_count, 'length_group': paired.length_group})
pred['error_type'] = np.where((y == 0) & (s >= 0), 'FP', np.where((y == 1) & (s < 0), 'FN', 'correct'))
pred.to_csv(OUT / 'validation_predictions.csv', index=False)
errors = pd.concat([pred[pred.error_type == 'FP'].nlargest(5, 'p_ai_uncalibrated'),
                    pred[pred.error_type == 'FN'].nsmallest(5, 'p_ai_uncalibrated')])
errors.to_csv(OUT / 'errors_to_review.csv', index=False)
notes_path = TEAM / 'neural_error_notes.json'
if notes_path.exists():
    notes = json.loads(notes_path.read_text())
    errors['observed_features'] = [notes.get(str(i), {}).get('observed', 'REQUIRES TEXT REVIEW') for i in errors.id]
    errors['possible_reason'] = [notes.get(str(i), {}).get('possible_reason', 'REQUIRES TEXT REVIEW') for i in errors.id]
    errors.to_csv(OUT / 'error_analysis.csv', index=False)
(OUT / 'evaluation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
