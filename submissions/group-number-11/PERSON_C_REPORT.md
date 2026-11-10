# Person C — Q2 and Q4

Status: methodology draft only. Competition train.csv was unavailable during
implementation; no experiment results or qualitative findings are claimed here.

## Q2. Effect of training data size

We use the team's committed 8,000/2,000 training/validation IDs. The model is
word-level (1–2 grams) plus character-level (3–5 grams) TF-IDF with Linear SVM,
C=0.5. Both vectorizers use min_df=2 and sublinear term frequency, with vocabulary
caps of 100,000 word features and 150,000 character features. All settings remain
fixed across training sizes.

Within the fixed training partition, we independently shuffle each class with
seed 42 and interleave the class pools to obtain balanced, nested subsets of
500, 1,000, 2,000, 5,000, and 8,000 examples. TF-IDF vocabulary and IDF weights
are fitted anew on each subset. The same 2,000 validation examples are evaluated
at every size. Training AUROC is measured on the actual training subset; both
training and validation AUROC use SVM decision scores.

Pending actual run:

- Insert `person_c_results/learning_curve_results.csv` and `learning_curve.png`.
- Describe the training/validation gap and its change with sample size.
- Quantify the validation improvement from 5,000 to 8,000 examples before
  judging whether further labeled data might help.
- Discuss that one seeded curve cannot establish uncertainty or prove a plateau.

## Q4. Generalization and error analysis

We evaluate the selected configuration fitted on the complete fixed training
partition. The training/validation AUROC gap measures the difference between
in-sample ranking performance and held-out ranking performance; a high training
AUROC alone does not establish poor generalization.

Text length is measured as the number of whitespace-separated words. The 1/3
and 2/3 quantiles of training lengths define three right-closed groups. We report
validation AUROC, total count, class counts, and FP/FN counts for each group.
Length associations are descriptive and may reflect topic or class differences.

For error analysis, human is label 0 and AI is label 1. A positive SVM margin
predicts AI. False positives are human passages predicted as AI; false negatives
are AI passages predicted as human. We select the five highest-margin false
positives and the five lowest-margin false negatives. This selection emphasizes
confident failures rather than estimating the prevalence of particular patterns.

To report P(AI), we fit a sigmoid to three-fold out-of-fold SVM margins from the
training partition. Each fold independently fits its TF-IDF representation.
The sigmoid is applied to the full-training model's validation margins; the
validation labels are never used for calibration. The original SVM threshold
is retained, and its corresponding probability threshold is reported. These
estimated probabilities may still be imperfectly calibrated because the
cross-validation models train on less data than the final model.

Pending actual run and case inspection:

- Insert overall training/validation AUROC and the confusion matrix.
- Insert the group boundaries and subgroup performance table.
- Read all ten selected passages locally; create `error_analysis.csv` containing
  ID, error type, P(AI), and passage-specific possible failure reasons.
- Summarize recurring observed patterns, distinguish hypotheses from evidence,
  and propose a targeted improvement. No additional Q5 experiment is required.
- Discuss the team's reported public/validation performance gap without treating
  different evaluation populations as directly comparable estimates or claiming
  an unverified cause.

## Reproducibility and disclosure

See the team README for the exact command. The script writes split and input
hashes, software versions, parameters, and runtimes. OpenAI Codex assisted with
implementation and this methodology draft. Actual metrics, the plot, and
case-specific interpretations remain to be checked once the data is available.
