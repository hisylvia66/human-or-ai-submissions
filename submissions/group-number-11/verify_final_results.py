"""Verify saved experiment integrity and the exact locally generated submission."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, confusion_matrix
from run_person_c_evaluation import load_fixed_split

TEAM = Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=Path,required=True)
    ap.add_argument('--submission',type=Path,default=TEAM/'artifacts/submission.csv')
    args=ap.parse_args()
    tr,va=load_fixed_split(args.data_dir/'train.csv')
    final=TEAM/'final_results'
    selected=json.loads((final/'selected_model.json').read_text())
    summary=json.loads((final/'evaluation_summary.json').read_text())
    comparison=pd.read_csv(final/'refinement_results.csv')
    winner=comparison.sort_values(['validation_auroc','fit_seconds'],ascending=[False,True]).iloc[0]
    assert winner.config==selected['config']==summary['config']
    assert len(comparison)==selected['n_candidates']==6
    scores=pd.read_csv(final/'validation_predictions.csv')
    assert scores.id.tolist()==va.id.tolist() and np.array_equal(scores.label,va.label)
    assert len(scores)==2000 and scores.id.is_unique
    assert np.isclose(roc_auc_score(scores.label,scores.decision_score),selected['validation_auroc'],atol=1e-12)
    assert np.isclose(roc_auc_score(scores.label,scores.p_ai),selected['validation_auroc'],atol=1e-12)
    assert np.allclose(scores.p_ai,expit(summary['calibration_slope']*scores.decision_score+summary['calibration_intercept']))
    assert np.array_equal(scores.predicted_label,(scores.decision_score>0).astype(int))
    assert confusion_matrix(scores.label,scores.predicted_label).tolist()==summary['confusion_matrix']
    assert np.isclose(expit(summary['calibration_intercept']),summary['classification_threshold_p_ai'])
    original_scores=pd.read_csv(final/'refinement_validation_scores.csv')
    assert original_scores.id.tolist()==va.id.tolist()
    assert np.allclose(scores.decision_score,original_scores[selected['config']])
    groups=pd.read_csv(final/'length_group_results.csv')
    assert groups.n.sum()==2000
    for r in groups.itertuples():
        g=scores[scores.length_group==r.length_group]
        assert len(g)==r.n and g.label.sum()==r.n_ai
        assert np.isclose(roc_auc_score(g.label,g.decision_score),r.validation_auroc)
        assert (g.error_type=='FP').sum()==r.false_positives
        assert (g.error_type=='FN').sum()==r.false_negatives
    errors=pd.read_csv(final/'error_analysis.csv')
    assert errors.error_type.value_counts().to_dict()=={'FP':5,'FN':5}
    assert errors.possible_reason.notna().all()
    assert not errors.possible_reason.str.contains('REQUIRES TEXT REVIEW').any()
    notes=json.loads((TEAM/'final_error_notes.json').read_text())
    for r in errors.itertuples():
        assert (r.label,r.predicted_label)==({'FP':(0,1),'FN':(1,0)}[r.error_type])
        assert notes[str(r.id)]['possible_reason']==r.possible_reason
    members=pd.read_csv(TEAM/'person_c_results/learning_curve_subset_ids.csv')
    previous=set()
    for n in (500,1000,2000,5000,8000):
        ids=members.loc[members.n_train==n,'id']
        assert len(ids)==n and ids.is_unique and previous<=set(ids)<=set(tr.id)
        assert not set(ids)&set(va.id)
        assert tr.set_index('id').loc[ids].label.value_counts().to_dict()=={0:n//2,1:n//2}
        previous=set(ids)
    submission=pd.read_csv(args.submission)
    template=pd.read_csv(args.data_dir/'sample_submission.csv')
    assert submission.columns.tolist()==template.columns.tolist()==['id','label']
    assert len(submission)==3000 and submission.id.is_unique and submission.id.tolist()==template.id.tolist()
    assert submission.notna().all().all() and np.isfinite(submission.label).all()
    assert submission.label.between(0,1).all()
    manifest=json.loads((final/'submission_manifest.json').read_text())
    assert manifest['config']==selected['config'] and manifest['n_labeled']==10000
    assert hashlib.sha256(args.submission.read_bytes()).hexdigest()==manifest['prediction_sha256']
    assert hashlib.sha256((final/'selected_model.json').read_bytes()).hexdigest()==manifest['selection_sha256']
    for name,digest in manifest['input_sha256'].items():
        assert hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()==digest
    for f in final.glob('*.csv'):
        assert 'text' not in pd.read_csv(f,nrows=0).columns, f'Raw text in {f}'
    print('PASS: selection, fixed split, nested training subsets, calibration, metrics, groups, 10 reviewed errors, submission IDs/probabilities/hashes, text-free outputs.')


if __name__=='__main__':
    main()
