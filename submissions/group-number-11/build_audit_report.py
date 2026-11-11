"""Generate the post-submission audit appendix from saved experiment outputs."""
from pathlib import Path
import json
import pandas as pd

TEAM=Path(__file__).resolve().parent
out=TEAM/'audit_results';report=TEAM/'report'
s=pd.read_csv(out/'robustness_audit_summary.csv').set_index('model')
i=pd.read_csv(out/'iid_control_results.csv').groupby('model').auroc.mean()
e=json.loads((out/'candidate_evaluation.json').read_text())
manifest=json.loads((out/'candidate_submission_manifest.json').read_text())
public=manifest.get('public_score')
score_status=(f'The blend has confirmed public AUROC {public:.5f}; private AUROC remains unknown.'
              if public is not None else 'The current blend prediction file has no confirmed Kaggle score.')
stress=pd.read_csv(out/'candidate_stress_results.csv').set_index('condition').auroc
rows=[]
for name,label in [('raw_nb','Submitted NB model'),('style','Style-only model'),('raw_style25','75:25 NB/style blend')]:
 rows.append(f'{label} & {s.loc[name,"fixed_validation"]:.6f} & {i[name]:.6f} & {s.loc[name,"topic_mean"]:.6f} '+r'\\')
tex=r'''\newpage
\begingroup
\small
\setlength{\emergencystretch}{2em}
\section*{Post-submission audit and improved submission}
\textbf{Status.} Q0--Q4 describe the submitted NB model (public AUROC 0.85994). This appendix records a subsequent independent review and the NB/style blend. '''+score_status+r'''

\subsection*{What the review found}
The high fixed-validation score did not establish generalization. Among the supplied labels, no human passage contains double newlines, while 3,301 of 5,000 AI passages do. Many large NB coefficients involve line breaks or template markers. Applying Unicode/quote/whitespace normalization to both validation classes reduces NB AUROC from 0.977012 to 0.917921; also removing punctuation gives 0.888169. These tests demonstrate sensitivity, not the exact cause of hidden-test errors.

An independent three-fold classifier distinguishing training text from test text achieves AUROC 0.730335, confirming observable distribution differences without test labels. Submission IDs and ranks were verified; the increasing sigmoid preserves AUROC. No exact cross-split text duplicates were found. Removing three validation examples with training-neighbor similarity at least 0.9 leaves AUROC 0.976943, insufficient to explain the public-score gap.

\subsection*{Independent generalization diagnostics}
Using only the official 8,000 training texts, unigram TF-IDF, 48-dimensional SVD and K-means form 12 label-free text clusters. Three-fold GroupKFold holds complete clusters out. These are inferred text groups, not known hidden domains. An ordinary stratified three-fold control uses a similar training size. Neither diagnostic changes the official 2,000-row validation IDs.

\begin{tabularx}{\linewidth}{@{}Yrrr@{}}\toprule
Model & Fixed val. & Random CV & Cluster CV\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabularx}\par
\smallnote{CV values are unweighted means of three fold AUROCs. These are exploratory diagnostics; grouped folds differ in size and composition.}

\subsection*{New model and evidence limits}
The content-light style model uses normalized text, function-word proportions, lexical diversity, word/sentence-length distributions, punctuation/character statistics and sentence repetition. It is a histogram gradient-boosting classifier (220 iterations, learning rate 0.06, 15 leaves, minimum leaf size 30, L2 penalty 10). Each component margin is divided by its own training-score standard deviation; the selected blend uses 75\% NB and 25\% style. We compared four components and three fixed blends. Among candidates retaining the old fixed-validation AUROC, this blend has the best mean cluster-CV AUROC and improves every cluster fold. Selection is exploratory and reuses validation results; it is not preregistered confirmation.

'''+f'The blend reaches fixed-validation AUROC {e["validation_auroc"]:.6f}, with {e["confusion_matrix"][0][1]} false positives and {e["confusion_matrix"][1][0]} false negatives at margin zero. Its normalized-text stress AUROC is {stress["format_normalized"]:.6f}; residual sensitivity remains. '+r'''It is refitted on all 10,000 supplied labels, with full-pipeline three-fold calibration and no external labels. The separate file is \path{artifacts/submission_robust_blend.csv}. See \texttt{PROJECT\_REVIEW.md} for findings and reproduction commands. The public score remains far below fixed-validation AUROC; improved validation alone does not resolve the distribution gap.
'''
(report/'audit_appendix.tex').write_text(tex + '\\endgroup\n')
print('Generated audit appendix.')
