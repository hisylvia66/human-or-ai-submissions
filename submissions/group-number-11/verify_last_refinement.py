"""Independent checks for the final robustness experiment and chosen submission."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
p=argparse.ArgumentParser();p.add_argument('--data_dir',type=Path,required=True);a=p.parse_args()
T=Path(__file__).resolve().parent;O=T/'last_refinement_results'
y=pd.read_csv(a.data_dir/'train.csv').set_index('id').label
ids=pd.read_csv(T/'validation_ids.csv').id
summary=json.loads((O/'validate_summary.json').read_text())
assert summary['n_train']==8000 and summary['starting_checkpoint']=='artifacts/neural/validate/best'
table=pd.read_csv(O/'comparison.csv')
for condition in table.condition.unique():
    scores=pd.read_csv(O/f'validation_{condition}.csv')
    assert scores.id.is_unique and set(scores.id)<=set(ids)
    assert len(scores)==(2000 if condition=='clean' else 1000)
    assert np.array_equal(scores.label,y.loc[scores.id])
    assert np.isfinite(scores[['old_score','new_score']].to_numpy()).all()
    for model,values in [('baseline',scores.old_score),('new_only',scores.new_score),('equal_old_new_logit_mean',.5*(scores.old_score+scores.new_score))]:
        saved=table[(table.model==model)&(table.condition==condition)].auroc.item()
        assert np.isclose(saved,roc_auc_score(scores.label,values),rtol=0,atol=1e-12)
for condition in table.condition.unique():
    path=O/f'validation_256_{condition}.csv'
    if path.exists():
        scores=pd.read_csv(path)
        assert scores.id.is_unique and set(scores.id)<=set(ids) and np.array_equal(scores.label,y.loc[scores.id])
        for kind,values in [('new_only_256',scores.new_score),('equal_old_new_logit_mean_256',.5*(scores.new_score+scores.old_score))]:
            saved=table[(table.model==kind)&(table.condition==condition)].auroc.item()
            assert np.isclose(saved,roc_auc_score(scores.label,values),rtol=0,atol=1e-12)
wide=table.pivot(index='model',columns='condition',values='auroc');eligible=wide[wide.clean>=summary.get('clean_floor',.985)]
expected=eligible.mean(axis=1).idxmax() if len(eligible) else wide.clean.idxmax()
assert expected==summary['chosen']
if (O/'refit_summary.json').exists():
    fit=json.loads((O/'refit_summary.json').read_text())
    assert fit['n_train']==10000 and fit['starting_checkpoint']=='artifacts/neural/refit/final' and fit['chosen']==expected
    assert fit['effective_batch_size']==32 and fit['schedule_batches']==500
    assert 0<fit['continuation_batches']<=500
    assert fit['continuation_rows_seen']==fit['continuation_batches']*fit['microbatch_size']
    assert np.isclose(fit['continuation_epochs'],fit['continuation_batches']/np.ceil(10000/fit['microbatch_size']),rtol=0,atol=1e-12)
    for key in ['seed','learning_rate','label_smoothing','max_length','inference']:assert fit[key]==summary[key],key
    for name,sha in fit['input_sha256'].items():assert hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest()==sha
    path=T/'artifacts/submission_final_robust_neural.csv';pred=pd.read_csv(path);sample=pd.read_csv(a.data_dir/'sample_submission.csv')
    assert list(pred.columns)==list(sample.columns) and pred.id.equals(sample.id)
    assert len(pred)==3000 and pred.id.is_unique and pred.notna().all().all() and pred.label.between(0,1).all()
    raw=pd.read_csv(T/'artifacts/last_refinement/test_scores.csv').set_index('id').loc[pred.id]
    want=raw.new_score if expected.startswith('new_only') else (.5*(raw.new_score+raw.old_score) if expected.startswith('equal_old_new_logit_mean') else raw.old_score)
    assert np.allclose(raw.selected_score,want,rtol=0,atol=1e-12)
    assert np.allclose(pred.label,expit(want),rtol=0,atol=1e-14)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==fit['prediction_sha256']
for manifest_name,filename in [('refit_summary.json','submission_deberta.csv'),('rank_blend_submission_manifest.json','submission_deberta_robust_rank50.csv')]:
    previous=json.loads((T/'neural_results'/manifest_name).read_text())
    assert hashlib.sha256((T/'artifacts'/filename).read_bytes()).hexdigest()==previous['prediction_sha256']
print('PASS: fixed holdout IDs, stress labels/counts, all AUROCs, recorded adaptive selection, separate checkpoint lineage; available final IDs, scores, transforms and hashes.')
