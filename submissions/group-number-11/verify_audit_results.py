"""Verify independent diagnostic coverage, blend arithmetic, calibration and candidate IDs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,confusion_matrix

TEAM=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data_dir',type=Path,required=True);a=ap.parse_args()
    out=TEAM/'audit_results'
    tr=set(pd.read_csv(TEAM/'train_ids.csv').id);va=pd.read_csv(TEAM/'validation_ids.csv').id
    scores=pd.read_csv(out/'robustness_audit_scores.csv');summary=pd.read_csv(out/'robustness_audit_summary.csv').set_index('model')
    groups=pd.read_csv(out/'training_topic_clusters.csv').set_index('id').topic_cluster
    seen_ids=set();seen_clusters=set()
    for condition,g in scores.groupby('condition'):
        if condition=='fixed_validation':assert g.id.tolist()==va.tolist()
        else:
            ids=set(g.id);clusters=set(groups.loc[g.id]);assert not seen_ids&ids and not seen_clusters&clusters
            assert ids<=tr;seen_ids|=ids;seen_clusters|=clusters
        assert np.allclose(g.raw_style25,.75*g.raw_nb+.25*g['style'])
        assert np.allclose(g.aug_style25,.75*g.augmented_nb+.25*g['style'])
        assert np.allclose(g.aug_style50,.5*g.augmented_nb+.5*g['style'])
        for name in summary.index:assert np.isclose(roc_auc_score(g.label,g[name]),summary.loc[name,condition],atol=1e-12)
    assert seen_ids==tr and not seen_ids&set(va)
    control=pd.read_csv(out/'iid_control_scores.csv')
    assert control.id.is_unique and set(control.id)==tr
    manifest=json.loads((out/'candidate_submission_manifest.json').read_text())
    evaluation=json.loads((out/'candidate_evaluation.json').read_text())
    eligible=summary[summary.fixed_validation>=.977012-1e-12]
    assert eligible.topic_mean.idxmax()==manifest['config']==evaluation['config']
    val=pd.read_csv(out/'candidate_validation_predictions.csv')
    assert val.id.tolist()==va.tolist()
    assert np.isclose(roc_auc_score(val.label,val.p_ai),evaluation['validation_auroc'])
    assert np.allclose(val.p_ai,expit(evaluation['calibration_slope']*val.decision_score+evaluation['calibration_intercept']))
    assert confusion_matrix(val.label,val.predicted_label).tolist()==evaluation['confusion_matrix']
    errors=pd.read_csv(out/'candidate_errors.csv');assert errors.error_type.value_counts().to_dict()=={'FP':5,'FN':5}
    sub_path=TEAM/'artifacts/submission_robust_blend.csv';sub=pd.read_csv(sub_path);template=pd.read_csv(a.data_dir/'sample_submission.csv')
    assert sub.columns.tolist()==template.columns.tolist()==['id','label']
    assert len(sub)==3000 and sub.id.is_unique and sub.id.tolist()==template.id.tolist()
    assert sub.notna().all().all() and sub.label.between(0,1).all() and np.isfinite(sub.label).all()
    assert hashlib.sha256(sub_path.read_bytes()).hexdigest()==manifest['prediction_sha256']
    for file,digest in manifest['input_sha256'].items():assert hashlib.sha256((a.data_dir/file).read_bytes()).hexdigest()==digest
    assert json.loads((TEAM/'final_results/submission_manifest.json').read_text())['public_score']==.85994
    print('PASS: official IDs, disjoint topic groups, OOF coverage, blend arithmetic, all AUROCs, calibration, error types, separate submission IDs/hashes, preserved actual score.')

if __name__=='__main__':main()
