"""One fixed 50:50 rank blend with the latest submitted model; no weight search.

Ranks are computed separately within each prediction set and use no labels.
Outputs are ranking scores in (0, 1), not calibrated P(AI). Validation is reused
for choosing among the neural-only model and this single fixed blend.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from submission_records import write_manifest

TEAM=Path(__file__).resolve().parent
OUT=TEAM/'neural_results'
summary=json.loads((OUT/'validate_summary.json').read_text())
neural=pd.read_csv(OUT/f"validate_epoch_{summary['best_epoch']}_scores.csv")
old=pd.read_csv(TEAM/'audit_results/candidate_validation_predictions.csv')
val=neural.merge(old,on='id',validate='one_to_one',suffixes=('_neural','_old'))
assert len(val)==2000 and (val.label_neural==val.label_old).all()
def blend(a,b):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return .5*(rankdata(a)+rankdata(b))/(len(a)+1)
s=blend(val.decision_score_neural,val.decision_score_old)
rows=[{'model':'submitted_robust_blend','validation_auroc':roc_auc_score(val.label_neural,val.decision_score_old)},
      {'model':'deberta','validation_auroc':roc_auc_score(val.label_neural,val.decision_score_neural)},
      {'model':'deberta_robust_rank50','validation_auroc':roc_auc_score(val.label_neural,s)}]
pd.DataFrame(rows).to_csv(OUT/'candidate_comparison.csv',index=False)
rng=np.random.default_rng(42);deltas=[]
for _ in range(1000):
    idx=rng.integers(0,len(val),len(val));sample=val.iloc[idx]
    if sample.label_neural.nunique()==2:
        deltas.append(roc_auc_score(sample.label_neural,s[idx])-roc_auc_score(sample.label_neural,sample.decision_score_old))
(OUT/'rank_blend_evaluation.json').write_text(json.dumps({
    'validation_auroc':rows[-1]['validation_auroc'],
    'delta_vs_submitted_blend':rows[-1]['validation_auroc']-rows[0]['validation_auroc'],
    'paired_bootstrap_95ci_conditional_on_selection':np.quantile(deltas,[.025,.975]).tolist(),
    'limitations':'Exploratory reused holdout; bootstrap keeps fitted empirical-rank transform fixed. No public score inferred.'
},indent=2)+'\n')
pd.DataFrame({'id':val.id,'label':val.label_neural,'ranking_score':s}).to_csv(OUT/'rank_blend_validation.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
test_path=TEAM/'artifacts/deberta_test_scores.csv'
if test_path.exists():
    manifest=json.loads((OUT/'refit_summary.json').read_text())
    assert manifest['revision']==summary['revision'] and manifest['best_epoch']==summary['best_epoch']
    assert hashlib.sha256(test_path.read_bytes()).hexdigest()==manifest['test_scores_sha256']
    newtest=pd.read_csv(test_path)
    oldtest=pd.read_csv(TEAM/'artifacts/submission_robust_blend.csv')
    old_manifest=json.loads((TEAM/'audit_results/candidate_submission_manifest.json').read_text())
    assert hashlib.sha256((TEAM/'artifacts/submission_robust_blend.csv').read_bytes()).hexdigest()==old_manifest['prediction_sha256']
    assert newtest.id.is_unique and oldtest.id.is_unique and set(newtest.id)==set(oldtest.id)
    test=oldtest.merge(newtest,on='id',validate='one_to_one')
    score=blend(test.decision_score,test.label)
    submission=pd.DataFrame({'id':test.id,'label':score})
    assert len(submission)==3000 and submission.notna().all().all() and submission.label.between(0,1).all()
    path=TEAM/'artifacts/submission_deberta_robust_rank50.csv'
    submission.to_csv(path,index=False)
    write_manifest(OUT/'rank_blend_submission_manifest.json',{
        'model':'deberta_robust_rank50','neural_weight':.5,'robust_weight':.5,
        'score_type':'Mean empirical percentile rank, not calibrated probability',
        'validation_auroc':rows[-1]['validation_auroc'],
        'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'input_sha256':manifest['input_sha256'],
        'component_prediction_sha256':{'neural':manifest['prediction_sha256'],'robust':old_manifest['prediction_sha256']},
        'public_score':None,'private_score':None})
    print('Saved',path)
