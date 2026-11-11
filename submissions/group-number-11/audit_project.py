"""Independent raw-data and shortcut audit. Test text is used only in aggregate diagnostics."""
import argparse,json,hashlib,re,unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from final_model import TextModel

TEAM=Path(__file__).resolve().parent

def stats(text):
    words=re.findall(r'\b\w+\b',text.lower());n=max(1,len(words));chars=max(1,len(text))
    return dict(words=n,chars=chars,newlines=text.count('\n'),double_newlines=text.count('\n\n'),
                ascii_quotes=(text.count('"')+text.count("'"))/chars,
                curly_quotes=sum(text.count(c) for c in '“”‘’')/chars,
                non_ascii=sum(ord(c)>127 for c in text)/chars,
                digits=sum(c.isdigit() for c in text)/chars,
                punctuation=sum(unicodedata.category(c).startswith('P') for c in text)/chars,
                unique_word_fraction=len(set(words))/n,
                mean_word_length=sum(map(len,words))/n,
                repeated_sentence_fraction=1-len(set(re.split(r'[.!?\n]+',text.lower())))/max(1,len(re.split(r'[.!?\n]+',text.lower()))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir',type=Path,required=True);a=ap.parse_args()
    out=TEAM/'audit_results';out.mkdir(exist_ok=True)
    full=pd.read_csv(a.data_dir/'train.csv');un=pd.read_csv(a.data_dir/'unlabeled.csv');test=pd.read_csv(a.data_dir/'test.csv')
    tr_ids=pd.read_csv(TEAM/'train_ids.csv').id;va_ids=pd.read_csv(TEAM/'validation_ids.csv').id
    # Reconstruct independently, without importing any existing split helper.
    assert full.id.is_unique and len(full)==10000
    assert len(tr_ids)==8000 and len(va_ids)==2000 and not set(tr_ids)&set(va_ids)
    assert set(tr_ids)|set(va_ids)==set(full.id)
    tr=full.set_index('id').loc[tr_ids].reset_index();va=full.set_index('id').loc[va_ids].reset_index()
    records=[]
    for name,frame in [('train',tr),('validation',va),('unlabeled',un),('test',test)]:
        features=pd.DataFrame([stats(t) for t in frame.text]);features['id']=frame.id.to_numpy()
        features['set']=name
        if 'label' in frame:features['label']=frame.label.to_numpy()
        records.append(features)
    all_stats=pd.concat(records,ignore_index=True)
    all_stats.to_csv(out/'observable_features.csv',index=False)
    aggregates=all_stats.groupby(['set','label'],dropna=False).mean(numeric_only=True).drop(columns='id')
    aggregates.to_csv(out/'aggregate_text_statistics.csv')
    print(aggregates.to_string(),flush=True)
    features=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True)
    X=features.fit_transform(tr.text);V=features.transform(va.text)
    distances,indices=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute').fit(X).kneighbors(V)
    near=pd.DataFrame({'validation_id':va.id,'train_id':tr.id.to_numpy()[indices[:,0]],'cosine_similarity':1-distances[:,0],
                       'validation_label':va.label,'train_label':tr.label.to_numpy()[indices[:,0]]})
    near.sort_values('cosine_similarity',ascending=False).head(30).to_csv(out/'nearest_training_passages.csv',index=False)
    print('Validation nearest-neighbor counts:',{str(t):int((near.cosine_similarity>=t).sum()) for t in [.8,.9,.95,.99]},flush=True)
    domain=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=40000,sublinear_tf=True)),
                     ('clf',LogisticRegression(C=1,max_iter=1000))])
    y=np.r_[np.zeros(len(tr)),np.ones(len(test))]
    ds=cross_val_predict(domain,pd.concat([tr.text,test.text]),y,cv=StratifiedKFold(3,shuffle=True,random_state=42),method='predict_proba')[:,1]
    domain_auc=roc_auc_score(y,ds)
    print('Independent train/test text-only domain-classifier AUROC:',domain_auc,flush=True)
    pd.DataFrame({'id':np.r_[tr.id,test.id],'set':['train']*len(tr)+['test']*len(test),'domain_test_probability':ds}).to_csv(out/'domain_scores.csv',index=False)
    # Inspect training-derived feature weights, not test passages or test labels.
    model=TextModel('nbsvm').fit(tr.text,tr.label)
    names=model.vectorizer.get_feature_names_out();effective=model.clf.coef_[0]*model.ratio
    ix=np.r_[np.argsort(effective)[:40],np.argsort(effective)[-40:][::-1]]
    pd.DataFrame({'feature':[repr(names[i]) for i in ix],'effective_weight':effective[ix]}).to_csv(out/'strongest_model_features.csv',index=False)
    normal=lambda s:s.str.lower().str.split().str.join(' ')
    summary={'train_validation_exact_overlap':len(set(tr.text)&set(va.text)),
             'train_validation_normalized_overlap':len(set(normal(tr.text))&set(normal(va.text))),
             'validation_neighbor_cosine_ge_09':int((near.cosine_similarity>=.9).sum()),
             'train_test_domain_cv_auroc':domain_auc,
             'observed_public_auroc':.85994,'validation_auroc':.977012,
             'validation_minus_public':.977012-.85994,
             'caveat':'Different evaluation populations; domain classifier does not identify the cause of label prediction failures.'}
    keep = ~va.id.isin(set(near.loc[near.cosine_similarity >= .9, 'validation_id']))
    summary['validation_auroc_excluding_3_high_similarity_rows'] = roc_auc_score(va.label[keep], model.decision_function(va.text[keep]))
    rng = np.random.default_rng(123)
    val_scores = model.decision_function(va.text)
    subset_aucs = []
    for _ in range(1000):
        ids = rng.choice(len(va), 900, replace=False)
        subset_aucs.append(roc_auc_score(va.label.iloc[ids], val_scores[ids]))
    sub = pd.read_csv(TEAM/'artifacts/submission.csv')
    sampling = dict(same_distribution_900row_auroc_95interval=np.quantile(subset_aucs,[.025,.975]).tolist(),
                    submission_unique_probabilities=int(sub.label.nunique()),
                    submission_min_probability=float(sub.label.min()),submission_max_probability=float(sub.label.max()),
                    note='Descriptive resampling of reused validation data, not a confidence interval for the Kaggle population.')
    (out/'submission_and_sampling_audit.json').write_text(json.dumps(sampling,indent=2)+'\n')
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
