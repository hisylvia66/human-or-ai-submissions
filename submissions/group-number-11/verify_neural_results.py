"""Recompute recorded neural metrics and check local submissions/provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
p=argparse.ArgumentParser();p.add_argument('--data_dir',type=Path,required=True);a=p.parse_args()
T=Path(__file__).resolve().parent;O=T/'neural_results'
source=json.loads((O/'pretrained_source.json').read_text())
summary=json.loads((O/'validate_summary.json').read_text())
labels=pd.read_csv(a.data_dir/'train.csv').set_index('id').label
tr=pd.read_csv(T/'train_ids.csv').id;va=pd.read_csv(T/'validation_ids.csv').id
assert len(tr)==8000 and len(va)==2000 and not set(tr)&set(va)
assert set(tr)|set(va)==set(labels.index)
assert summary['revision']==source['revision']
epochs=pd.read_csv(O/'validate_epochs.csv')
assert summary['best_epoch']==int(epochs.loc[epochs.validation_auroc.idxmax(),'epoch'])
for r in epochs.itertuples():
    f=pd.read_csv(O/f'validate_epoch_{r.epoch}_scores.csv')
    assert f.id.is_unique and set(f.id)==set(va)
    assert np.array_equal(f.label,labels.loc[f.id]) and np.isfinite(f.decision_score).all()
    assert np.isclose(roc_auc_score(f.label,f.decision_score),r.validation_auroc,atol=1e-12,rtol=0)
rank_path=O/'rank_blend_validation.csv'
if rank_path.exists():
    rank=pd.read_csv(rank_path)
    best=pd.read_csv(O/f"validate_epoch_{summary['best_epoch']}_scores.csv")
    old=pd.read_csv(T/'audit_results/candidate_validation_predictions.csv')
    paired=rank.merge(best[['id','decision_score']],on='id',validate='one_to_one').merge(old[['id','decision_score']],on='id',suffixes=('_neural','_old'),validate='one_to_one')
    expected=.5*(rankdata(paired.decision_score_neural)+rankdata(paired.decision_score_old))/(len(paired)+1)
    assert set(paired.id)==set(va) and np.allclose(paired.ranking_score,expected,atol=1e-14)
    evaluation=json.loads((O/'rank_blend_evaluation.json').read_text())
    assert np.isclose(roc_auc_score(paired.label,paired.ranking_score),evaluation['validation_auroc'],atol=1e-12,rtol=0)
if (O/'diagnostics.json').exists():
    d=json.loads((O/'diagnostics.json').read_text())
    train=pd.read_csv(O/'training_predictions.csv')
    assert train.id.is_unique and set(train.id)==set(tr) and np.array_equal(train.label,labels.loc[train.id])
    assert np.isclose(roc_auc_score(train.label,train.decision_score),d['train_auroc'],atol=1e-12,rtol=0)
    for condition in ('format_normalized','punctuation_removed'):
        stress=pd.read_csv(O/f'{condition}_scores.csv')
        assert stress.id.is_unique and set(stress.id)==set(va) and np.array_equal(stress.label,labels.loc[stress.id])
        assert np.isclose(roc_auc_score(stress.label,stress.decision_score),d[f'{condition}_validation_auroc'],atol=1e-12,rtol=0)
refit_path=O/'refit_summary.json'
if refit_path.exists():
    refit=json.loads(refit_path.read_text());assert refit['n_train']==10000 and refit['n_validation']==0
    for key in ['best_epoch','revision','learning_rate','batch_size','accumulation','max_length','precision','freeze_embeddings','seed']:
        assert refit[key]==summary[key],key
    for name,sha in refit['input_sha256'].items():assert hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest()==sha
    sample=pd.read_csv(a.data_dir/'sample_submission.csv')
    pred=pd.read_csv(T/'artifacts/submission_deberta.csv')
    score=pd.read_csv(T/'artifacts/deberta_test_scores.csv')
    assert list(pred.columns)==list(sample.columns) and pred.id.equals(sample.id)
    assert pred.id.is_unique and len(pred)==3000 and pred.label.between(0,1).all()
    merged=pred.merge(score,on='id',validate='one_to_one')
    assert np.allclose(merged.label,expit(merged.decision_score),atol=1e-14)
    assert hashlib.sha256((T/'artifacts/submission_deberta.csv').read_bytes()).hexdigest()==refit['prediction_sha256']
    assert hashlib.sha256((T/'artifacts/deberta_test_scores.csv').read_bytes()).hexdigest()==refit['test_scores_sha256']
    blend_path=T/'artifacts/submission_deberta_robust_rank50.csv'
    if blend_path.exists():
        blend=pd.read_csv(blend_path);old=pd.read_csv(T/'artifacts/submission_robust_blend.csv')
        check=blend.merge(old,on='id',suffixes=('_blend','_old'),validate='one_to_one').merge(score,on='id',validate='one_to_one')
        expected=.5*(rankdata(check.label_old)+rankdata(check.decision_score))/(len(check)+1)
        assert blend.id.equals(sample.id) and np.allclose(check.label_blend,expected,atol=1e-14)
        m=json.loads((O/'rank_blend_submission_manifest.json').read_text())
        assert hashlib.sha256(blend_path.read_bytes()).hexdigest()==m['prediction_sha256']
print('PASS: fixed IDs, epoch selection, independently recomputed AUROCs, model revision; available refit settings, prediction transforms, ID order and hashes.')
