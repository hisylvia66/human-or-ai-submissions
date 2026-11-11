"""Similar-training-size random-fold control for the topic-held-out audit."""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from robust_models import RobustModel

TEAM=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir',type=Path,required=True);a=ap.parse_args()
    d=pd.read_csv(a.data_dir/'train.csv').set_index('id').loc[pd.read_csv(TEAM/'train_ids.csv').id].reset_index()
    out=TEAM/'audit_results';rows=[];preds=[]
    for fold,(fit,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(d,d.label),1):
        s=pd.DataFrame({'id':d.id.iloc[held].to_numpy(),'label':d.label.iloc[held].to_numpy(),'fold':fold})
        for kind in ('raw_nb','style'):
            print(f'Random-fold control {fold}: {kind}',flush=True)
            m=RobustModel(kind).fit(d.text.iloc[fit],d.label.iloc[fit]);s[kind]=m.decision_function(d.text.iloc[held])
        s['raw_style25']=.75*s.raw_nb+.25*s['style']
        for kind in ('raw_nb','style','raw_style25'):
            r={'fold':fold,'model':kind,'n_fit':len(fit),'n_held':len(held),'auroc':roc_auc_score(s.label,s[kind])};rows.append(r);print(r,flush=True)
        preds.append(s)
        pd.DataFrame(rows).to_csv(out/'iid_control_results.csv',index=False)
        pd.concat(preds).to_csv(out/'iid_control_scores.csv',index=False)
    clusters=pd.read_csv(out/'training_topic_clusters.csv').set_index('id').loc[d.id].topic_cluster.to_numpy()
    v=TfidfVectorizer(stop_words='english',min_df=5,max_features=12000);X=v.fit_transform(d.text);names=v.get_feature_names_out()
    groups=[]
    for c in sorted(set(clusters)):
        mask=clusters==c;s=np.asarray(X[mask].mean(axis=0)).ravel();top=names[np.argsort(s)[-10:][::-1]]
        groups.append({'cluster':int(c),'n':int(mask.sum()),'ai_rate':float(d.label[mask].mean()),'top_terms':', '.join(top)})
    pd.DataFrame(groups).to_csv(out/'topic_cluster_descriptions.csv',index=False)

if __name__=='__main__':main()
