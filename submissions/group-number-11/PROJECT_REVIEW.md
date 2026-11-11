# Post-submission project review

The submitted NB-weighted model achieved **0.85994 public AUROC**, despite **0.977012** on the fixed validation set. This review starts from the supplied CSVs and reconstructs split membership independently. It preserves the submitted model and its measured score rather than overwriting them with an unsubmitted candidate.

## Findings, ranked by practical importance

### 1. The representation is highly sensitive to formatting

Among all 5,000 human-labeled passages, none contain double newlines or Markdown headings; 3,301 AI-labeled passages contain double newlines and 102 contain headings. Several of the submitted model's largest effective coefficients involve line breaks, Markdown marks and recurring template phrases. These can encode source/preprocessing artifacts rather than general authorship evidence.

A controlled stress test on **both classes** reduces the submitted model's AUROC from **0.977012 to 0.917921** after Unicode/quote/whitespace normalization; additionally removing punctuation reduces it to **0.888169**. This demonstrates sensitivity to these transformations. It does not prove which changes actually caused errors in Kaggle's unlabeled test data.

### 2. Random held-out validation overstates performance on different text groups

A fresh text-only domain classifier distinguishes official training text from Kaggle test text with three-fold AUROC **0.730335**. This detects distribution differences without test labels; it is not an authorship detector or proof of the direction of shift.

Within the official 8,000 training rows, label-free unigram TF-IDF/SVD/K-means yields 12 text clusters. These approximate topic and source/style groups, not known hidden domains or generators. One small cluster dominated by HTML-like `br` tokens has 165 human labels and one AI label, illustrating source artifacts. Three-fold GroupKFold holds complete clusters out. A separate stratified three-fold control uses a similar training-set size:

| Model | Mean ordinary-fold AUROC | Mean text-cluster-held-out AUROC |
| --- | ---: | ---: |
| Submitted NB model | 0.969897 | 0.903318 |
| Normalized stylometric model | 0.955965 | 0.895069 |
| Fixed 75% NB + 25% stylometric score blend | **0.984046** | **0.945723** |

The blend improves all three text-cluster folds: **0.929200 / 0.960873 / 0.947096**, versus **0.862515 / 0.923679 / 0.923761** for NB alone. This is stronger internal evidence than another gain on the reused random validation split, but it still does not establish private-test performance. Grouped folds differ in class/topic composition and are not exactly equal in size to random folds. These diagnostics do not replace the mandatory fixed 8k/2k comparisons.

### 3. Submission alignment and sigmoid conversion are not the apparent failure

The generated submission has exactly 3,000 template-aligned IDs, finite probabilities in [0,1], and 3,000 distinct score values. Dataset and submission hashes match the saved run. The calibration sigmoid is increasing, so it preserves ranking; it cannot by itself repair or explain this AUROC loss through a reversed class mapping or collapsed ranks.

As a descriptive check, repeatedly taking 900 rows from the existing validation set gives a central 95% AUROC range of **0.971250–0.982726**, far above 0.85994. This is not a confidence interval for Kaggle's different population. The exact public subset composition remains unknown.

### 4. Near duplicates are a limitation, not a sufficient explanation here

No raw or case/whitespace-normalized exact train/validation duplicates were found. Three validation rows have a training nearest neighbor with word-TF-IDF cosine similarity at least 0.9. Excluding those rows changes NB AUROC only to **0.976943**. This check does not prove absence of semantic near duplicates, but these three rows do not explain the gap.

### 5. Existing calibration, selection and reporting limitations

- Earlier Q3 code shares a fitted training vocabulary across calibration folds, and later rounds place pseudo-labels derived from all training targets into every calibration fold. That can make inner calibration optimistic. The official held-out validation set still remains excluded from fitting; this is not proof of validation-label leakage.
- The initial refinement selected from six candidates solely by the reused validation AUROC. Its bootstrap interval was already labeled conditional and not selection-adjusted; it should not be interpreted as a guarantee of test improvement.
- Historical and current-environment scores differ slightly. The report preserves their provenance; it should not silently combine them as one exact run.
- The old `generate_submission.py` is positional and sigmoid-only; current submission routes explicitly join by ID and use full-pipeline training-only calibration. The current submitted CSV was verified using the safer route.
- The old report was too optimistic as a stopping point: high random-validation AUROC did not resolve robustness and distribution-shift concerns. The actual public score is now included.

## Independent candidates and selection

The new code compares raw NB, normalized NB, original-plus-normalized augmentation, and a content-light stylometric histogram-gradient-boosting model. The latter uses normalized text, function-word proportions, word/sentence-length distributions, lexical diversity, punctuation/character statistics and sentence repetition. It uses no pretrained detector, external labels, hidden test labels, or external LLM test predictions.

Each component margin is divided by its own **training-score** standard deviation. Three blend weights were fixed in the code: raw/style 75:25, augmented/style 75:25, and augmented/style 50:50. No weight was tuned on public leaderboard scores.

| Candidate | Fixed validation AUROC | Mean text-cluster-held-out AUROC |
| --- | ---: | ---: |
| Raw NB | 0.977012 | 0.903318 |
| Normalized NB | 0.962252 | 0.873088 |
| Augmented NB | 0.967228 | 0.881165 |
| Style only | 0.958415 | 0.895069 |
| Raw NB + style 25% | **0.988136** | **0.945723** |
| Augmented NB + style 25% | 0.984598 | 0.934315 |
| Augmented NB + style 50% | 0.983282 | 0.932587 |

Selection prefers the highest mean cluster-held-out score among candidates whose fixed-validation AUROC is at least 0.977012. This rule was recorded after viewing fixed-validation results but before the cluster comparisons finished; this is an exploratory audit, not a preregistered confirmation study. It selects **raw_style25**. Normalization or augmentation alone was not adopted merely because it sounded more robust.

## Reproduce and next submission

From this team directory with the existing pinned dependencies:

```bash
python audit_project.py --data_dir data/competition
python run_robustness_audit.py --data_dir data/competition
python run_iid_audit_control.py --data_dir data/competition
python generate_audit_candidate.py --data_dir data/competition
python verify_audit_results.py --data_dir data/competition
```

The separate candidate is `artifacts/submission_robust_blend.csv`. It is refitted on all 10,000 supplied labeled examples, with a fresh full-pipeline three-fold out-of-fold sigmoid. It uses no unlabeled pseudo-labels. The previous `artifacts/submission.csv` and its 0.85994 public score remain unchanged.

The integrated report keeps the latest **actually submitted** NB model in Q0/Q4 and adds this exploratory audit as an appendix. Do not assign the old score to the blend. Once the candidate is submitted and its score is confirmed, synchronize final selection, Q4, reproduction commands and the report's score table together. No public/private improvement is claimed for the candidate yet.
