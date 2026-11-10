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
