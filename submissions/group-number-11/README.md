# Group Number 11 — Human or AI?

## Team Members
- Yilin Gao / yg695
- Yingzi Gu / yg667
- Ansh Raj Bhargava / ab3396

## Kaggle Team
Group Number 11

## Baseline
Initial baseline: TF-IDF + Logistic Regression

Validation AUROC: 0.9343
Training AUROC: 0.9953

## Reproducibility
Use the fixed train/validation split provided in:
- `train_ids.csv`
- `validation_ids.csv`

Random seed: 42

## Run
Install dependencies:

pip install -r requirements.txt

Run baseline:

python run_baseline.py

## Current Final Model

Our current selected model is a Linear SVM with combined word-level and character-level TF-IDF features.

- Word TF-IDF: unigrams and bigrams
- Character TF-IDF: 3-5 grams
- Linear SVM: `C=0.5`
- Random seed: `42`
- Fixed validation split: 8,000 training / 2,000 validation examples

Validation AUROC: approximately `0.9711`

Kaggle public AUROC for this model: `0.87240`

### Generate the Kaggle submission

Place the Kaggle files locally under `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Then run:

```bash
python generate_submission.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_submission data/sample_submission.csv \
  --out submission.csv
```

The script retrains the selected model on all 10,000 labeled training examples and outputs AI-authorship scores in `[0,1]`.

### Notes

The Kaggle competition files are not committed to this repository.
Model selection and hyperparameter tuning were based on the fixed held-out validation set, not Kaggle test labels.


## Person B experiments (Q3 + Q5)

All Person B scripts use the committed split (`train_ids.csv` / `validation_ids.csv`, seed 42),
the selected model (word 1–2 + char 3–5 TF-IDF + LinearSVC, C=0.5) and seed 42.
`data_utils.py` finds the data automatically: `./data`, then `/kaggle/input/...` (Kaggle notebook),
then `kagglehub`, then the Kaggle API (Colab / laptop with `KAGGLE_USERNAME` + `KAGGLE_KEY`).

**One-click run:** open `person_b_experiments.ipynb` on Kaggle with the competition attached as
input and Internet on → Run All (about 10–15 min CPU). Results land in `/kaggle/working/personB_results/`.

### Q3 — unlabeled data
```bash
python run_unlabeled_experiment.py --data_dir data
```
Self-training: SVM margins → P(AI) via Platt scaling on out-of-fold training scores; unlabeled rows
with P(AI) ≥ t or ≤ 1−t are pseudo-labeled and added, t ∈ {0.99, 0.95, 0.90, 0.80, all}; then
3 iterative rounds at the best t. Also: TF-IDF fit on train+unlabeled text, simulated pseudo-label
error rate, train/unlabeled/test domain-shift AUROC, model confidence per set.
Outputs: `unlabeled_experiment_results.csv`, `unlabeled_pseudo_label_noise.csv`,
`unlabeled_domain_shift.csv`, `unlabeled_confidence_by_set.csv`, `unlabeled_self_training.png`.

### Q5 — robustness to adversarially edited AI text
```bash
python run_robustness_improvement.py --data_dir data --submit R2
```
R0 word-only SVM (reference) · R1 word+char SVM (baseline) · R2 R1 + unicode normalization (NFKC,
homoglyph folding, zero-width removal) · R3 R2 + typo augmentation of both classes. Evaluated on
clean validation and on copies where only AI passages get typos / Cyrillic homoglyphs / zero-width
characters, with paired-bootstrap 95% CIs vs R1, plus a census of such characters in each split.
`--submit R2` retrains R2 on all 10,000 labeled rows and writes `submission_R2.csv`.
Outputs: `robustness_improvement_results.csv`, `robustness_char_census.csv`,
`robustness_validation_scores.csv`.

## Person C experiments (Q2 + Q4)

Run from this team directory (Python 3.10+ and the existing requirements):

```bash
python run_person_c_evaluation.py \
  --train_csv /path/to/competition/train.csv \
  --review_text data/person_c_error_review.csv
```

Only the labeled competition file is needed. The script reads the committed ID
files directly and rejects missing, duplicate, overlapping, or mismatched IDs.
It never creates another train/validation split. Results go to `person_c_results/`.

- Q2 uses balanced, nested subsets of 500/1,000/2,000/5,000/8,000 rows (seed 42).
  Each size independently fits word (1–2) and character (3–5) TF-IDF and
  LinearSVC(C=0.5), with exactly the selected model's remaining parameters.
  Outputs: `learning_curve_results.csv`, `learning_curve_subset_ids.csv`, and
  `learning_curve.png` / `.pdf`. Fit time includes feature fitting and SVM training.
- Q4 uses the 8,000-row model from Q2. Length is whitespace-separated word count;
  training-set tertiles define short/medium/long groups. Outputs:
  `length_group_results.csv`, `validation_predictions.csv`,
  `selected_error_predictions.csv`, `error_analysis.csv`, and `evaluation_summary.json`.
  Reviewed case annotations are stored in `person_c_error_notes.json`; if a rerun
  selects a different ID, its explanation is marked `REQUIRES TEXT REVIEW`.
- AUROC and FP/FN use the original SVM margins (classification threshold 0).
  P(AI) comes from a logistic sigmoid fitted to three-fold out-of-fold training
  margins, refitting the full feature pipeline per fold. This preserves the
  selected model's ranking. The equivalent probability threshold is recorded in
  the summary and need not equal 0.5. Calibration reliability is not guaranteed.
- The selected cases are the five most confidently wrong predictions per class;
  they illustrate failure modes and are not a representative sample of all errors.
  Read the optional local review file to write case-specific explanations.
  That file contains competition text, is ignored by Git, and must not be committed.
- One seeded learning curve is descriptive; it does not estimate variation across
  training subsets. Length groups can differ in topic and class composition.

The script records software versions, input/split hashes, group boundaries, and
calibration parameters. `PERSON_C_REPORT.md` contains the completed Q2/Q4 report,
including all ten case explanations and limitations. The committed run used
Python 3.14.2; `requirements-person-c.txt` pins its dependency versions and can
be installed instead of the general requirements for closer reproduction.

Validation AUROC at 500/1,000/2,000/5,000/8,000 examples was
0.890672 / 0.923734 / 0.947738 / 0.965055 / 0.971228; training AUROC was 1.0
throughout. Short/medium/long validation AUROC was 0.958732 / 0.973797 / 0.983390.
The full model made 50 false positives and 127 false negatives at margin 0.

The original `run_word_char_improvement.py` also produced 0.971228 in this
environment, matching the new script exactly. This differs by 0.000106 from
the historical committed result 0.971122; the precise historical cause is not
confirmed. Neither the validation split nor model configuration was changed.
To reproduce that cross-check (the original script checks all four C values):

```bash
python run_word_char_improvement.py \
  --train_csv /path/to/competition/train.csv \
  --out person_c_results/original_script_reproduction.csv
```

### AI assistance for Person C

OpenAI Codex assisted with Q2/Q4 implementation, experiment execution, inspection
of the ten selected passages, qualitative hypotheses, and report drafting.
Split-integrity checks, a synthetic smoke check, output consistency checks, and
an independent rerun of the original model script were performed. The plot was
visually inspected. Grace should review the interpretations and disclosure before
including them in the final report; these checks do not claim human review.
