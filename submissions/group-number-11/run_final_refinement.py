"""Compare a finite candidate set on committed IDs, then select by validation AUROC.

No test text or labels are accessed. All candidates are recorded, including failures
 to improve. Confidence intervals are descriptive and not selection-adjusted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from final_model import CONFIGS, TextModel, SEED
from run_person_c_evaluation import load_fixed_split

TEAM = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, default=TEAM / 'final_results')
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tr, va = load_fixed_split(args.data_dir / 'train.csv')
    un = pd.read_csv(args.data_dir / 'unlabeled.csv')
    if un.text.isna().any():
        raise ValueError('Unlabeled text is missing')
    # Exact normalized duplicates are diagnosed, never silently removed from fixed split.
    norm = lambda texts: texts.str.lower().str.split().str.join(' ')
    train_texts, val_texts, un_texts = map(norm, (tr.text, va.text, un.text))
    audit = dict(train_validation_exact_overlap=int(va.text.isin(set(tr.text)).sum()),
                 train_validation_normalized_overlap=int(val_texts.isin(set(train_texts)).sum()),
                 unlabeled_validation_normalized_overlap=int(un_texts.isin(set(val_texts)).sum()),
                 within_training_normalized_duplicates=int(train_texts.duplicated().sum()),
                 near_duplicate_check='Not performed; normalized exact-match check only')
    # Exclude duplicate held-out passages from pseudo-label input, without using their labels.
    un = un.loc[~un_texts.isin(set(val_texts))].copy()
    (out / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    scores = pd.DataFrame({'id': va.id, 'label': va.label})
    rows = []
    for config, description in CONFIGS.items():
        print(f'Fitting {config}: {description}', flush=True)
        start = time.perf_counter()
        model = TextModel(config).fit(tr.text, tr.label, un.text)
        fit_s = time.perf_counter() - start
        train_s = model.decision_function(tr.text)
        val_s = model.decision_function(va.text)
        scores[config] = val_s
        row = dict(config=config, description=description,
                   train_auroc=roc_auc_score(tr.label, train_s),
                   validation_auroc=roc_auc_score(va.label, val_s), fit_seconds=fit_s,
                   pseudo_rounds=json.dumps(model.rounds))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'refinement_results.csv', index=False)
        scores.to_csv(out / 'refinement_validation_scores.csv', index=False)
        print(json.dumps(row), flush=True)
    result = pd.DataFrame(rows)
    best = result.sort_values(['validation_auroc', 'fit_seconds'], ascending=[False, True]).iloc[0]
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(1000):
        i = rng.integers(0, len(va), len(va))
        y = va.label.to_numpy()[i]
        if len(np.unique(y)) == 2:
            deltas.append(roc_auc_score(y, scores[best.config].to_numpy()[i]) -
                          roc_auc_score(y, scores.baseline.to_numpy()[i]))
    selection = dict(config=best.config, selection_rule='Highest fixed-validation AUROC; fit time breaks exact ties',
                     validation_auroc=float(best.validation_auroc),
                     delta_vs_baseline=float(best.validation_auroc - result.iloc[0].validation_auroc),
                     paired_bootstrap_95ci=np.quantile(deltas, [.025, .975]).tolist(),
                     ci_caveat='Descriptive, conditional on selected candidate; not adjusted for model selection',
                     n_candidates=len(rows), seed=SEED, train_csv_sha256=digest(args.data_dir / 'train.csv'),
                     unlabeled_csv_sha256=digest(args.data_dir / 'unlabeled.csv'),
                     test_used_for_selection=False)
    (out / 'selected_model.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
