# Local neural experiment

Before neural development, the confirmed public AUROC was **0.88628** for
`artifacts/submission_robust_blend.csv`, compared with its fixed-validation
AUROC of 0.988136. A higher fixed-validation score cannot establish a public
score of 0.98–0.99. The neural experiment tests a different representation;
no hidden labels or externally labeled detection dataset are used.

## Model and protocol

- General-purpose [Microsoft DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small),
  revision `a36c739020e01763fe789b4b85e2df55d6180012`.
  See the [DeBERTaV3 paper](https://arxiv.org/abs/2111.09543).
- All inference and supervised fine-tuning run locally. Pretraining is external
  general-language training and must be disclosed separately from supplied labels.
- Train only on committed 8,000 training IDs; choose between two epoch checkpoints
  by AUROC on the same committed 2,000 validation IDs. This is reused validation,
  not an independent final test. Test text is used only at final inference.
- Normalize whitespace, truncate to first 256 tokens, freeze embeddings, fine-tune
  transformer encoder and newly initialized classification head. Truncation may
  discard useful later text. Measured tokenization shows 51.55% of training and
  50.5% of validation rows exceed 256 tokens (9.39% / 8.4% exceed 384). No claim of calibrated softmax probabilities.
- AdamW LR 3e-5, weight decay 0.01, batch 16, accumulation 2, 10% linear warmup,
  linear decay, gradient norm clipped at 1, seed 42, two epochs, bfloat16.
- Apple MPS local acceleration. GPU nondeterminism and different environments can
  cause small reproduction differences. Model weights/cache stay ignored.
- Final refit inherits the selected validation configuration and epoch count,
  trains from the original pretrained weights on all 10,000 supplied labels.

## Reproduction (from this team directory)

Use a Python environment compatible with the pinned dependencies. The recorded
run uses Python 3.14.2, PyTorch 2.10.0 and Transformers 4.57.6.

```bash
python -m pip install -r requirements-neural.txt
python download_pretrained.py
python train_neural.py --data_dir data/competition --mode validate \
  --epochs 2 --batch_size 16 --accumulation 2 --max_length 256 \
  --learning_rate 0.00003 --precision bf16 --freeze_embeddings
python evaluate_neural.py
python diagnose_neural.py --data_dir data/competition
python compare_neural_candidates.py
python train_neural.py --data_dir data/competition --mode refit
python compare_neural_candidates.py
python verify_neural_results.py --data_dir data/competition
```

Optional `--dependency_dir` appends an isolated dependency directory to Python's
search path. It is an environment convenience, not part of the model settings.
CPU fallback is available; it may be much slower. Use `--precision fp32` on a
backend that does not support bfloat16, and record that changed configuration.

`neural_results/` stores pinned provenance, epoch scores and metrics. Generated
weights and candidate `artifacts/submission_deberta.csv` remain local and ignored.
A generated candidate has **no Kaggle score until actually submitted**. Existing
submission files and their confirmed leaderboard scores are preserved.

A single fixed 50:50 empirical-rank blend with the submitted NB/style model is
also compared on validation, with no weight sweep. Ranks use no labels and are
computed separately on validation and test prediction sets. The resulting
`submission_deberta_robust_rank50.csv` contains ranking scores, not calibrated
probabilities. Selection remains exploratory on a reused holdout.

## Measured validation results

| Model | Fixed-validation AUROC | Kaggle public AUROC |
|---|---:|---:|
| Previously submitted NB/style blend | 0.988136 | 0.88628 |
| DeBERTa epoch 1 | 0.987573 | Not submitted |
| DeBERTa epoch 2 (selected; full-data refit) | 0.989423 | 0.92433 |
| Selected DeBERTa + fixed 50:50 rank blend | 0.994594 | 0.92164 |

The neural-versus-previous-blend AUROC gain is 0.001287, with a descriptive paired
bootstrap interval [-0.003095, 0.005289]. The rank blend's gain is 0.006458,
interval [0.004089, 0.008688]. Both intervals condition on prior checkpoint/model
selection and do not establish improvement on the shifted Kaggle distribution.
The rank-blend bootstrap holds its fitted empirical-rank transform fixed.

At softmax 0.5, pure DeBERTa has 204 FP / 18 FN and 88.9% validation accuracy.
High ranking performance therefore does not imply calibrated probabilities or an
appropriate operating threshold. `neural_error_notes.json` records observations
and tentative explanations for five confident FP and five confident FN; no label
corrections or causal attribution are claimed. Q2's controlled learning curve and
Q3's original self-training experiments remain historical comparisons.

The measured validation training run took 21.2 minutes. Full-data refit plus
3,000-row test inference took 30.3 minutes on this local MPS run. Both candidate
CSV files were generated and verified by ID, range, arithmetic and SHA-256.

## Post-selection diagnostics

Reloading the selected checkpoint exactly reproduces validation AUROC 0.989423.
Training AUROC on its 8,000 training rows is 0.995930 (gap 0.006507). Applying
Unicode/quote/whitespace normalization to both validation classes yields 0.989299;
normalizing and removing punctuation yields 0.980220. These are post-selection
synthetic stress tests, not additional candidates or measured Kaggle conditions.
The final submissions keep the validated whitespace-only preprocessing. Diagnostic
inference took 6.8 minutes; no further model fitting was performed.

Both scores above are confirmed by the user screenshots. The neural-only public
score exceeds the rank blend despite the reverse validation ordering. A final
local robustness continuation is documented in `FINAL_REFINEMENT.md`.

Final selection: pure DeBERTa, public AUROC **0.92433**. The subsequent robustness continuation scored **0.91085** and was rejected. Public scores influenced this selection; private performance remains unknown.
