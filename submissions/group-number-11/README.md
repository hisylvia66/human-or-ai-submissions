# Group 11 — Human or AI?

| Team member | NetID | Main contributions |
|---|---|---|
| Ansh Raj Bhargava | ab3396 | Q3 self-training and Q5 robustness experiments |
| Yilin Gao | yg695 | Q1 pipeline, model comparison, tuning and shared split |
| Yingzi Gu | yg667 | Q2/Q4, later model refinement, reproducibility and report integration |

## Final deliverables

- **Final model: DeBERTa-v3-small**, public AUROC **0.92433**; fixed-validation AUROC **0.989423**. Private score is unknown.
- [Final report PDF](report/report.pdf) · [LaTeX source](report/main.tex)
- [Team branch and folder](https://github.com/hisylvia66/human-or-ai-submissions/tree/submission/group-number-11/submissions/group-number-11)
- [Course submission PR #13](https://github.com/sekhari/human-or-ai-submissions/pull/13)
- Selected Kaggle file: `artifacts/submission_deberta.csv`, generated locally and excluded from Git. Its exact prediction hash and confirmed score are in [the manifest](neural_results/refit_summary.json).

All project changes are in `submissions/group-number-11/`, on branch `submission/group-number-11`. The repository preserves experiments and negative results; historical script names containing `final` refer to the earlier NB stage, not the selected neural model.

## Confirmed submissions

| File | Model | Kaggle public AUROC |
|---|---|---:|
| `submission_word_svm_c05.csv` | Word SVM | 0.85049 |
| `submission.csv` | NB-weighted word/character SVM | 0.85994 |
| `submission_robust_blend.csv` | NB/stylometry blend | 0.88628 |
| `submission_deberta_robust_rank50.csv` | Neural/classical rank blend | 0.92164 |
| **`submission_deberta.csv`** | **Pure DeBERTa (selected)** | **0.92433** |
| `submission_final_robust_neural.csv` | Robust neural continuation (rejected) | 0.91085 |

All scores above were confirmed by user-provided Kaggle screenshots. The earlier team-message score 0.87240 was not confirmed and is not used. Best-public selection reuses leaderboard feedback; neither public nor reused validation performance is an untouched estimate of private-test performance.

The rank blend had higher local validation AUROC (0.994594) but lower public performance than pure DeBERTa. The last continuation improved three synthetic stress tests but reduced public AUROC by 0.01348. Its final fitting budget was shorter than its development run; preprocessing, augmentation and context also changed. We do not attribute its failure to a single cause. Details: [neural experiment](NEURAL_EXPERIMENT.md), [rejected continuation](FINAL_REFINEMENT.md), [earlier distribution audit](PROJECT_REVIEW.md).

## Setup and selected-model reproduction

Run all commands from this directory. Place the supplied `train.csv`, `unlabeled.csv`, `test.csv` and `sample_submission.csv` in `data/competition/`. Do not commit data, full passage excerpts, credentials, caches or weights. The fixed split is committed as `train_ids.csv` (8,000) and `validation_ids.csv` (2,000); never create a replacement split.

The recorded neural environment is Python 3.14.2, PyTorch 2.10.0, Transformers 4.57.6, local Apple MPS and bfloat16. Dependencies are pinned in `requirements-neural.txt`. CPU fallback is slower; use FP32 if the backend lacks bfloat16 and record the changed configuration. GPU and environment differences can affect numerical reproduction.

```bash
python -m pip install -r requirements-neural.txt
python download_pretrained.py
python train_neural.py --data_dir data/competition --mode validate \
  --epochs 2 --batch_size 16 --accumulation 2 --max_length 256 \
  --learning_rate 0.00003 --precision bf16 --freeze_embeddings
python evaluate_neural.py
python diagnose_neural.py --data_dir data/competition
python train_neural.py --data_dir data/competition --mode refit
python verify_neural_results.py --data_dir data/competition
python export_selected_submission.py
```

The downloader pins general-language pretrained weights to revision `a36c739020e01763fe789b4b85e2df55d6180012`. Supervised fine-tuning uses only competition labels. Development and full-data refit start independently from the original pretrained model; the all-label checkpoint is never used to evaluate the overlapping validation rows. The chosen epoch count is two. Whitespace normalization, first-256-token truncation and uncalibrated softmax scores are used. No pseudo-labels or ensemble enter the final selection. See `NEURAL_EXPERIMENT.md` for exact optimizer settings, timings and limitations.

Reproduction creates `artifacts/submission_deberta.csv`. The export command verifies its manifest and writes the identical bytes to **`artifacts/final_submission/submission.csv`**, satisfying the course filename without overwriting the historical NB file. Confirmed scores belong to the recorded prediction hashes, not automatically to newly regenerated files; changed predictions archive prior scored manifests. No hidden labels or external prediction APIs are used.

## Required experiments and evidence

Q2 and Q3 deliberately retain classical configurations for controlled experiments; they are not neural learning curves or neural self-training results. Q4 in the final report evaluates the selected neural development checkpoint, including training/validation AUROC, three length groups and five FP/five FN with uncalibrated P(AI) and inspected hypotheses.

| Question/stage | Code | Recorded evidence |
|---|---|---|
| Q1 comparison/tuning | `run_model_comparison.py`, `run_hyperparameter_tuning.py` | `final_results/q1_*.csv` |
| Q2 fixed-model learning curve | `run_person_c_evaluation.py` | `person_c_results/`, nested subset IDs and plot |
| Q3 unlabeled data | `run_unlabeled_experiment.py` | `unlabeled_*` files, historical teammate run |
| Q4 selected DeBERTa | `evaluate_neural.py`, `diagnose_neural.py` | `neural_results/`, `neural_error_notes.json` |
| Q5 original robustness | `run_robustness_improvement.py` | `robustness_*` files |
| NB refinement | `run_final_refinement.py`, `evaluate_final_model.py` | `final_results/` |
| Distribution audit/blend | `run_robustness_audit.py`, `run_iid_audit_control.py` | `audit_results/` |
| Neural and fixed rank blend | `train_neural.py`, `compare_neural_candidates.py` | `neural_results/` |
| Rejected continuation | `run_last_refinement.py`, `select_last_refinement.py` | `last_refinement_results/` |

Classical reproduction uses `requirements-final.txt`:

```bash
python -m pip install -r requirements-final.txt
python run_model_comparison.py --train_csv data/competition/train.csv --out final_results/q1_model_comparison.csv
python run_hyperparameter_tuning.py --train_csv data/competition/train.csv --out final_results/q1_hyperparameter_tuning.csv
python run_person_c_evaluation.py --train_csv data/competition/train.csv --review_text data/person_c_error_review.csv
python run_final_refinement.py --data_dir data/competition
python evaluate_final_model.py --data_dir data/competition
python generate_final_submission.py --data_dir data/competition
python verify_final_results.py --data_dir data/competition
```

The last four commands reproduce the **historical NB model**, not the final selected Kaggle model. They preserve its metrics and Q4 comparisons. Q2 sizes are 500/1,000/2,000/5,000/8,000, with unchanged word/character SVM settings and the same validation IDs. Its nested subsets use seed 42; each refits TF-IDF. Q3/Q5 original experiments write to the working directory, so replay them separately:

```bash
mkdir -p artifacts/replay_q3 artifacts/replay_q5
(cd artifacts/replay_q3 && python ../../run_unlabeled_experiment.py --data_dir ../../data/competition)
(cd artifacts/replay_q5 && python ../../run_robustness_improvement.py --data_dir ../../data/competition)
```

Historical teammate versions were not pinned, so small numerical differences are possible. Person B's original calibration shares a training-only feature vocabulary across folds; the revised NB-stage self-training refits full pipelines per fold. The report distinguishes them. Initial scripts/notebooks and result files are retained as experimental history, not competing final entrypoints.

For the audit and rejected continuation, use `PROJECT_REVIEW.md` and `FINAL_REFINEMENT.md`. `compare_neural_candidates.py` additionally requires the historical blend predictions; it is optional for reproducing the selected pure model.

## Build PDF and standalone Overleaf project

```bash
python build_report.py
python build_audit_report.py
python build_neural_report.py
python build_last_refinement_report.py
tectonic --keep-logs --outdir artifacts/report report/main.tex
python package_report.py
```

Tectonic is a separate system dependency. The package command updates the versioned `report/report.pdf` and creates local `artifacts/group11_overleaf.zip`. Upload that ZIP to Overleaf, select `main.tex`, and compile with pdfLaTeX. Every table and figure is included. Do not upload the entire repository to Overleaf.

`report/status.json` records the selected model, all latest confirmed statuses and exact team names. The report retains historical required experiments but uses DeBERTa consistently in the final summary, Q4 and final prediction instructions. The rejected continuation's actual 0.91085 is included; private scores remain null.

## Verification and disclosure

The three existing verifiers (`verify_final_results.py`, `verify_neural_results.py`, `verify_last_refinement.py`) recompute saved metrics and check splits, selected checkpoints, submission IDs, transforms and hashes. `verify_audit_results.py` checks the earlier blend. Data-dependent verification requires local competition files; the repository is intentionally not a data distribution package.

Codex assisted Yingzi Gu with code, execution, debugging, passage inspection, hypotheses and LaTeX drafting. Experiments were run and outputs checked; automated verification is not a claim of human review. Team members remain responsible for reviewing their contributions and disclosures. Error explanations are hypotheses, not causal attributions. No exact normalized cross-split duplicates were found; near duplicates were not exhaustively checked. Repeated holdout and public leaderboard use limit generalization claims.
