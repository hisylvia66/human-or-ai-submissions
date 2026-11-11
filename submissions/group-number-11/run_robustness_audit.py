"""Independent review using fixed validation + training-only topic-held-out diagnostics.

This is a diagnostic, not a replacement for the official 8k/2k split. No test data
are read. Candidate blends are fixed before scores are examined.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from robust_models import RobustModel,normalize

TEAM=Path(__file__).resolve().parent
KINDS=['raw_nb','normalized_nb','augmented_nb','style']

def extend_blends(scores):
    scores['raw_style25']=.75*scores['raw_nb']+.25*scores['style']
    scores['aug_style25']=.75*scores['augmented_nb']+.25*scores['style']
    scores['aug_style50']=.5*scores['augmented_nb']+.5*scores['style']
    return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir',type=Path,required=True);a=ap.parse_args()
    out=TEAM/'audit_results';out.mkdir(exist_ok=True)
    data=pd.read_csv(a.data_dir/'train.csv').set_index('id')
    tr=data.loc[pd.read_csv(TEAM/'train_ids.csv').id].reset_index()
    va=data.loc[pd.read_csv(TEAM/'validation_ids.csv').id].reset_index()
    # Unsupervised topic grouping uses only official training texts, never validation/test.
    T=TfidfVectorizer(ngram_range=(1,1),stop_words='english',min_df=5,max_features=12000).fit_transform(tr.text)
    Z=TruncatedSVD(n_components=48,random_state=42).fit_transform(T)
    norms=np.linalg.norm(Z,axis=1,keepdims=True);Z=Z/np.maximum(norms,1e-9)
    clusters=MiniBatchKMeans(n_clusters=12,n_init=10,random_state=42,batch_size=512).fit_predict(Z)
    pd.DataFrame({'id':tr.id,'topic_cluster':clusters,'label':tr.label}).to_csv(out/'training_topic_clusters.csv',index=False)
    rows=[];all_scores=[]
    def run(name,fit,held):
        scores=pd.DataFrame({'id':held.id.to_numpy(),'label':held.label.to_numpy(),'condition':name})
        for kind in KINDS:
            print(f'{name}: fitting {kind}',flush=True);start=time.perf_counter()
            model=RobustModel(kind).fit(fit.text,fit.label)
            scores[kind]=model.decision_function(held.text)
            if name=='fixed_validation':
                for condition,texts in [('format_normalized',[normalize(t) for t in held.text]),
                                        ('punctuation_removed',[__import__('re').sub(r'[^\w\s]',' ',normalize(t)) for t in held.text])]:
                    shifted=model.decision_function(texts)
                    rows.append({'condition':condition,'model':kind,'n':len(held),'auroc':roc_auc_score(held.label,shifted),'fit_and_score_seconds':None})
            rows.append({'condition':name,'model':kind,'n':len(held),'auroc':roc_auc_score(held.label,scores[kind]),'fit_and_score_seconds':time.perf_counter()-start})
            pd.DataFrame(rows).to_csv(out/'robustness_audit_results.csv',index=False)
            print(rows[-1],flush=True)
        scores=extend_blends(scores)
        for kind in ['raw_style25','aug_style25','aug_style50']:
            rows.append({'condition':name,'model':kind,'n':len(held),'auroc':roc_auc_score(held.label,scores[kind]),'fit_and_score_seconds':None})
            print(rows[-1],flush=True)
        all_scores.append(scores)
        pd.concat(all_scores).to_csv(out/'robustness_audit_scores.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'robustness_audit_results.csv',index=False)
    run('fixed_validation',tr,va)
    for fold,(fit,held) in enumerate(GroupKFold(n_splits=3).split(tr.text,tr.label,groups=clusters),1):
        run(f'topic_holdout_{fold}',tr.iloc[fit],tr.iloc[held])
    summary=pd.DataFrame(rows).pivot(index='model',columns='condition',values='auroc')
    summary['topic_mean']=summary[[c for c in summary if c.startswith('topic_holdout')]].mean(axis=1)
    summary.to_csv(out/'robustness_audit_summary.csv')
    print(summary.to_string(),flush=True)

if __name__=='__main__':main()
