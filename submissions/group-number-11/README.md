# Group Number 11 — Human or AI?

Yilin Gao (yg695), Yingzi Gu / Grace (yg667), Ansh Raj Bhargava (ab3396).
Work on `submission/group-number-11`. All project changes belong in this team directory.

Course draft PR: https://github.com/sekhari/human-or-ai-submissions/pull/13
The PR remains a draft pending team review; the new-model public score is now recorded.


## Post-submission independent review and next candidate

After the confirmed **0.85994** public score, an independent review found marked
format sensitivity and a large drop under training-only text-cluster holdout.
See [PROJECT_REVIEW.md](PROJECT_REVIEW.md) for evidence, code limitations and all
candidate results. The new fixed 75:25 NB/stylometry blend reaches **0.988136**
on official validation and **0.945723** mean cluster-CV AUROC (old NB: **0.903318**).
It improves all three held-out cluster folds, but has not yet been scored on Kaggle.

The **next candidate** is `artifacts/submission_robust_blend.csv`. The previous
`artifacts/submission.csv` remains the actually submitted NB model with 0.85994.
Do not confuse these files or transfer one model's score to the other.

```bash
python audit_project.py --data_dir data/competition
python run_robustness_audit.py --data_dir data/competition
python run_iid_audit_control.py --data_dir data/competition
python generate_audit_candidate.py --data_dir data/competition
python verify_audit_results.py --data_dir data/competition
python build_audit_report.py
```

The report includes a post-submission audit appendix; its main Q0/Q4 still describe
the model that was actually submitted. Promote the candidate and synchronize final
Q4 only after confirming the next submission decision. Observed scores are retained
only when prediction and input hashes match; changed predictions archive the old
manifest instead of inheriting its leaderboard score.

## Selected approach and submission status

The selected model is **NB-weighted word + character TF-IDF with LinearSVC(C=0.5)**.
It was selected from six recorded candidates using the committed 8,000/2,000 split.

- Final validation AUROC: **0.977012**; training AUROC: **0.999964**.
- Original word + character SVM baseline in the same environment: **0.971228**.
- Improvement: **0.005784**. The descriptive paired bootstrap interval is
  [0.001478, 0.010103]; it is conditional on selection, not multiplicity-adjusted.
- Final Kaggle predictions are generated locally at `artifacts/submission.csv`.
  The submitted new model achieved **public AUROC 0.85994**, confirmed by the user-provided Kaggle screenshot. Private AUROC remains unknown.
- The screenshot-confirmed previous public score is **0.85049**, for
  `submission_word_svm_c05.csv`. The earlier team message's 0.87240 is unverified
  and is not reported as the final model's score.

The NB weighting is inspired by [Wang and Manning (ACL 2012)](https://aclanthology.org/P12-2018/), adapted here to word/character TF-IDF.

The final model estimates smoothed class log-count ratios from binary training
feature occurrences, multiplies TF-IDF by those ratios, then fits the SVM. Word
features are 1–2 grams (100,000 cap); character features are 3–5 grams (150,000 cap).
Both use min_df=2, sublinear TF and default L2 normalization. Output probabilities
come from a logistic sigmoid fitted to full-pipeline 3-fold out-of-fold training
margins. Feature vocabularies, IDF and supervised weights are refit per fold.
The final Kaggle fit uses all 10,000 labeled rows. The selected model does not use
unlabeled examples, though Q3 tests their value.

## Install and data

The recorded environment is Python **3.14.2**. Exact direct dependencies are pinned:

```bash
cd submissions/group-number-11
python -m pip install -r requirements-final.txt
```

Place the supplied files in `data/competition/`:
`train.csv`, `unlabeled.csv`, `test.csv`, `sample_submission.csv`.
Do not commit competition data, full passage excerpts, credentials or model weights.
Existing `requirements.txt` is the unpinned dependency list for earlier experiments.

## Reproduce final selection, evaluation and predictions

Run from this team directory:

```bash
python run_final_refinement.py --data_dir data/competition
python evaluate_final_model.py --data_dir data/competition
python generate_final_submission.py --data_dir data/competition
python verify_final_results.py --data_dir data/competition
```

`final_model.py` is shared by all three workflows, so the evaluated and submitted
configurations are identical. `run_final_refinement.py` never reads test data.
It compares baseline, character weight 2, character-only, logistic regression C=5,
NB weighting and two-round self-training. Selection is highest validation AUROC,
with fit time breaking exact ties. Reusing the holdout for multiple experiments
can make the selected score optimistic. No public leaderboard result guided selection.

`final_results/` contains candidate metrics and scores, selected configuration,
input hashes, duplicate checks, final Q4 groups/errors, and the submission manifest.
`evaluate_final_model.py` verifies that its independently refitted margins match
those from selection. Error case annotations live in `final_error_notes.json`;
newly selected IDs are marked for text review rather than automatically invented.
The local review text is saved in ignored `data/final_error_review.csv`.

Final errors use SVM margin > 0, equivalent to calibrated P(AI) > **0.529366** in
this validation fit. The sigmoid preserves ranking and does not guarantee perfectly
calibrated probabilities. Five most-confident errors per class are inspected;
this purposive sample is not representative of all errors.

## Reproduce required experiments

### Q1: model comparison and tuning

These scripts now read the committed IDs instead of reconstructing a random split.
The original root-level result CSVs preserve earlier teammate runs. The report
uses the current-environment reruns under `final_results/` for Q1.

```bash
python run_model_comparison.py --train_csv data/competition/train.csv --out final_results/q1_model_comparison.csv
python run_hyperparameter_tuning.py --train_csv data/competition/train.csv --out final_results/q1_hyperparameter_tuning.csv
python run_word_char_improvement.py --train_csv data/competition/train.csv --out artifacts/word_char_replay.csv
```

The older `run_baseline.py` is retained as initial development history and takes
`--zip`; use the commands above for the final fixed-ID evaluation workflow.

### Q2: learning curve; historical supervised Q4

```bash
python run_person_c_evaluation.py --train_csv data/competition/train.csv --review_text data/person_c_error_review.csv
```

This preserves the original supervised word + character model at five balanced,
nested sample sizes (500/1,000/2,000/5,000/8,000; seed 42). Each size refits TF-IDF.
Validation AUROC is 0.890672 / 0.923734 / 0.947738 / 0.965055 / 0.971228.
The PNG/PDF plot, metrics and subset IDs are in `person_c_results/`.
Q2 is deliberately held fixed; final-model Q4 is separately rerun by
`evaluate_final_model.py`. `PERSON_C_REPORT.md` is historical baseline analysis;
use the integrated LaTeX report for the current final model.

### Q3/Q5: Person B's original experiments

The committed `unlabeled_*` and `robustness_*` files are Person B's recorded Kaggle
runs. To reproduce without replacing that historical evidence, run in an ignored
output directory (the scripts write several files to the working directory):

```bash
mkdir -p artifacts/replay_q3 artifacts/replay_q5
(cd artifacts/replay_q3 && python ../../run_unlabeled_experiment.py --data_dir ../../data/competition)
(cd artifacts/replay_q5 && python ../../run_robustness_improvement.py --data_dir ../../data/competition)
```

The report labels these runs separately because the historical dependency versions
were not pinned. Their baseline is 0.971122 versus the current 0.971228. Results may
vary slightly with software versions. The existing Kaggle/Colab notebook
`person_b_experiments.ipynb` is also retained.

Person B's calibration uses out-of-fold SVM margins with features shared within
the training partition. The final-refinement self-training candidate instead
refits full feature pipelines per fold and freezes its initial training-only
sigmoid across rounds. The two procedures are intentionally not presented as exact
replications. Both exclude the fixed validation set from fitting. Q3's historical
best was 0.972232; the revised candidate obtained 0.972186. Q5 normalization and
augmentation did not improve original-validation performance; negative results
are retained and explained. Synthetic robustness scores do not represent known
hidden-test subgroup scores.

`generate_submission.py` and `run_robustness_improvement.py --submit R2` are legacy
submission routes for earlier models. **Use `generate_final_submission.py` for
the currently selected model.**

## LaTeX report and Overleaf

`report/main.tex` follows every Q0–Q6 question in the supplied template. Its tables
and numeric macros are generated from result files, not manually copied scores.

```bash
python build_report.py
tectonic --keep-logs --outdir artifacts/report report/main.tex
python package_report.py
```

Tectonic is a separate system tool (not a pip dependency). On Overleaf, upload
`artifacts/group11_overleaf.zip`, select `main.tex`, and use pdfLaTeX. The package
script also copies the compiled PDF to `report/report.pdf`. The ZIP is standalone:
all tables and the learning curve are included, with no competition data.

After submitting `artifacts/submission.csv` to Kaggle, update the actual
`public_score` and `submission_status` in `report/status.json`, then regenerate,
compile and package the report. Leave private score null until it is revealed.
The GitHub PR URL is also stored in that status file. Never substitute an older
model's public score for the new model's score.

## Verification, history and disclosure

`verify_final_results.py` checks IDs, nested training-only subsets, final metrics,
calibration, subgroup counts, five FP/five FN, submission format and hashes.
No exact cross-split duplicates were found after case/whitespace normalization;
near duplicates were not exhaustively checked. Development history preserves the
original experiments and the later refinement rather than squashing them together.

Codex assisted Grace with implementation, execution, debugging, selected validation
passage inspection, hypotheses and report writing. The experiments were actually
run; metrics and the rendered PDF were checked. Automated verification is not
claimed as human review. Every member should review their contributions, claims
and AI-use disclosure before final submission. No external LLM API predicted test
labels and no hidden labels were retrieved.

The new submission is complete: public AUROC **0.85994**, an increase of **0.00945**
over the earlier word-only submission (0.85049). This compares different full
pipelines, not the isolated effect of NB weighting. Review the PDF and course PR,
then submit the PDF and PR URL in the course portal. Private performance remains
unknown until Kaggle reveals it.
