"""Train the selected approach on all labeled rows; write ID-aligned Kaggle probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from final_model import TextModel, oof_scores, sigmoid_calibrator

TEAM = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--selection',type=Path, default=TEAM/'final_results/selected_model.json')
    ap.add_argument('--out',type=Path, default=TEAM/'artifacts/submission.csv')
    args = ap.parse_args()
    selected = json.loads(args.selection.read_text())
    config = selected['config']
    tr, un, te, sample = [pd.read_csv(args.data_dir / name) for name in
                          ('train.csv','unlabeled.csv','test.csv','sample_submission.csv')]
    if len(tr)!=10000 or len(te)!=3000 or not tr.id.is_unique or not te.id.is_unique or not sample.id.is_unique:
        raise ValueError('Invalid competition rows/IDs')
    if set(te.id) != set(sample.id) or tr.text.isna().any() or te.text.isna().any():
        raise ValueError('Invalid test/template IDs or missing text')
    if hashlib.sha256((args.data_dir/'train.csv').read_bytes()).hexdigest() != selected['train_csv_sha256']:
        raise ValueError('Training data differs from model selection')
    start = time.perf_counter()
    print(f'Fitting {config} on all 10,000 labeled rows.',flush=True)
    model = TextModel(config).fit(tr.text,tr.label,un.text)
    print('Fitting out-of-fold probability calibration on labeled data only.',flush=True)
    calibrator = sigmoid_calibrator(oof_scores(config,tr.text,tr.label,un.text),tr.label)
    p = calibrator.predict_proba(model.decision_function(te.text).reshape(-1,1))[:,1]
    column = [c for c in sample.columns if c!='id']
    if len(column)!=1:
        raise ValueError('Expected one prediction column')
    pred = pd.DataFrame({'id':te.id,column[0]:p})
    sub = sample[['id']].merge(pred,on='id',how='left',validate='one_to_one')
    if sub.isna().any().any() or not np.isfinite(sub[column[0]]).all() or not sub[column[0]].between(0,1).all():
        raise ValueError('Invalid predictions')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sub.to_csv(args.out,index=False)
    manifest = dict(config=config,n_labeled=len(tr),n_unlabeled=len(un) if config=='selftrain2' else 0,
                    n_predictions=len(sub),training_and_calibration_seconds=time.perf_counter()-start,
                    calibration_slope=float(calibrator.coef_[0,0]),calibration_intercept=float(calibrator.intercept_[0]),
                    prediction_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                    selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                    input_sha256={name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                                  for name in ('train.csv','unlabeled.csv','test.csv','sample_submission.csv')},
                    public_score=None,private_score=None,kaggle_status='Generated locally; not submitted')
    # Manifest is safe to commit; prediction file remains an ignored local artifact.
    (TEAM/'final_results/submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {args.out}; {len(sub)} valid predictions.',flush=True)


if __name__=='__main__':
    main()
