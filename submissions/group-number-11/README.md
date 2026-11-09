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

