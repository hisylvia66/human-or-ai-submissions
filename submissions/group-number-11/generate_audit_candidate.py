"""Select the audit candidate, evaluate it, and create a separate Kaggle file.

Does not overwrite the previously submitted NB model or its public score.
Selection uses topic-held-out diagnostics among candidates retaining baseline
fixed-validation AUROC, as documented in audit_results/protocol.json.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from robust_models import BlendModel,normalize
from submission_records import write_manifest

TEAM=Path(__file__).resolve().parent

def sigmoid(scores,y):
    c=LogisticRegression(C=1e6,random_state=42).fit(np.asarray(scores).reshape(-1,1),y)
    if c.coef_[0,0]<=0:raise ValueError('Nonmonotonic calibration')
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir',type=Path,required=True);a=ap.parse_args()
    out=TEAM/'audit_results'
    summary=pd.read_csv(out/'robustness_audit_summary.csv').set_index('model')
    eligible=summary[summary.fixed_validation>=.977012-1e-12]
    chosen=eligible.topic_mean.idxmax()
    if chosen not in ['raw_style25','aug_style25','aug_style50']:raise ValueError('No supported blend improved the selection criterion')
    print('Selected audit candidate:',chosen,flush=True)
    full=pd.read_csv(a.data_dir/'train.csv');test=pd.read_csv(a.data_dir/'test.csv');sample=pd.read_csv(a.data_dir/'sample_submission.csv')
    tr=full.set_index('id').loc[pd.read_csv(TEAM/'train_ids.csv').id].reset_index()
    va=full.set_index('id').loc[pd.read_csv(TEAM/'validation_ids.csv').id].reset_index()
    start=time.perf_counter();m=BlendModel(chosen).fit(tr.text,tr.label)
    train_score=m.decision_function(tr.text);val_score=m.decision_function(va.text)
    expected=pd.read_csv(out/'robustness_audit_scores.csv');expected=expected[expected.condition=='fixed_validation'].set_index('id').loc[va.id,chosen]
    assert np.allclose(val_score,expected,atol=1e-9)
    control=pd.read_csv(out/'iid_control_scores.csv').set_index('id').loc[tr.id]
    if chosen not in control:raise ValueError('Run matching random-fold control before calibration')
    cal=sigmoid(control[chosen],tr.label)
    probabilities=cal.predict_proba(val_score.reshape(-1,1))[:,1]
    bounds=np.quantile(tr.text.str.split().str.len(),[1/3,2/3])
    val=pd.DataFrame({'id':va.id,'label':va.label,'decision_score':val_score,'p_ai':probabilities,
                      'predicted_label':(val_score>0).astype(int),'word_count':va.text.str.split().str.len()})
    val['length_group']=pd.cut(val.word_count,[-np.inf,*bounds,np.inf],labels=['short','medium','long'])
    val['error_type']=np.where((val.label==0)&(val.predicted_label==1),'FP',np.where((val.label==1)&(val.predicted_label==0),'FN','correct'))
    val.to_csv(out/'candidate_validation_predictions.csv',index=False)
    errors=pd.concat([val[val.error_type==kind].sort_values('decision_score',ascending=kind=='FN').head(5) for kind in ('FP','FN')])
    errors.to_csv(out/'candidate_errors.csv',index=False)
    errors.merge(va[['id','text']],on='id').to_csv(TEAM/'data/audit_candidate_errors.csv',index=False)
    groups=[]
    for name,g in val.groupby('length_group',observed=True):
        groups.append({'group':str(name),'n':len(g),'n_ai':int(g.label.sum()),'auroc':roc_auc_score(g.label,g.decision_score)})
    pd.DataFrame(groups).to_csv(out/'candidate_length_groups.csv',index=False)
    stress=[]
    import re
    for name,texts in [('format_normalized',[normalize(t) for t in va.text]),('punctuation_removed',[re.sub(r'[^\w\s]',' ',normalize(t)) for t in va.text])]:
        stress.append({'condition':name,'auroc':roc_auc_score(va.label,m.decision_function(texts))})
    pd.DataFrame(stress).to_csv(out/'candidate_stress_results.csv',index=False)
    review={'config':chosen,'train_auroc':roc_auc_score(tr.label,train_score),'validation_auroc':roc_auc_score(va.label,val_score),
            'topic_cv_mean_auroc':float(summary.loc[chosen,'topic_mean']),
            'calibration_slope':float(cal.coef_[0,0]),'calibration_intercept':float(cal.intercept_[0]),
            'probability_threshold_at_margin_zero':float(cal.predict_proba([[0]])[0,1]),
            'confusion_matrix':confusion_matrix(va.label,val.predicted_label).tolist(),
            'length_boundaries':bounds.tolist(),'public_score':None,'private_score':None,
            'selection_rule':'Best mean topic-held-out AUROC among candidates with fixed-validation AUROC >= 0.977012',
            'evaluation_seconds':time.perf_counter()-start}
    (out/'candidate_evaluation.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2),flush=True)
    print('Refitting on all 10,000 labeled rows and generating full-pipeline OOF calibration.',flush=True)
    model=BlendModel(chosen).fit(full.text,full.label)
    oof=np.zeros(len(full))
    for fold,(fit,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(full,full.label),1):
        print('Full-data calibration fold',fold,flush=True)
        cv=BlendModel(chosen).fit(full.text.iloc[fit],full.label.iloc[fit]);oof[held]=cv.decision_function(full.text.iloc[held])
    final_cal=sigmoid(oof,full.label)
    prob=final_cal.predict_proba(model.decision_function(test.text).reshape(-1,1))[:,1]
    prediction=pd.DataFrame({'id':test.id,'label':prob})
    assert test.id.is_unique and sample.id.is_unique and set(sample.id)==set(test.id)
    sub=sample[['id']].merge(prediction,on='id',validate='one_to_one',how='left')
    assert len(sub)==3000 and sub.notna().all().all() and sub.label.between(0,1).all()
    path=TEAM/'artifacts/submission_robust_blend.csv';sub.to_csv(path,index=False)
    manifest={'config':chosen,'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_labeled':len(full),'n_predictions':len(sub),
              'public_score':None,'private_score':None,'status':'Generated locally; not submitted',
              'calibration_slope':float(final_cal.coef_[0,0]),'calibration_intercept':float(final_cal.intercept_[0]),
              'input_sha256':{name:hashlib.sha256((a.data_dir/name).read_bytes()).hexdigest() for name in ('train.csv','test.csv','sample_submission.csv')}}
    write_manifest(out/'candidate_submission_manifest.json', manifest)
    print('Saved',path,flush=True)

if __name__=='__main__':main()
