# Group 11 report

Upload the standalone `group11_overleaf.zip` as a new Overleaf project. Set
`main.tex` as the main document and use pdfLaTeX (two passes for references).
All required tables and the learning-curve PDF are included.

The report follows Q0–Q6 in the course template, omitting the blank instructions
and rubric pages. No source passages or competition datasets are included.

The main report documents the historical NB stage: NB-weighted word/character TF-IDF + LinearSVC C=0.5,
validation AUROC 0.977012 and screenshot-confirmed Kaggle public AUROC 0.85994.
The audit appendix records the newer NB/style blend: validation 0.988136 and confirmed public score 0.88628. The neural appendix records validation AUROC 0.989423 (DeBERTa) and 0.994594 (fixed rank blend), with confirmed public scores 0.92433 and 0.92164, respectively. This is a working report, not yet the final model selection.
Private AUROC is not yet available. The older word-only model scored 0.85049.

In the repository, edit `report/status.json`, run `python build_report.py`, `python build_audit_report.py`, and
`python build_neural_report.py`, and `python build_last_refinement_report.py`, compile
`report/main.tex` and run `python package_report.py`. Tables and macros are derived
from saved experiments. In Overleaf alone, update the public/private score macros
and submission status in `metrics.tex` only after the matching submission is scored,
and sync those edits back to the repository's status JSON afterward.

Review the contribution table, AI-use disclosure, and interpretation with all team
members before course submission. The current disclosure does not assert that
human review has already occurred. Private score should remain unavailable until
Kaggle reveals it. The GitHub PR link must point to the course repository PR.
