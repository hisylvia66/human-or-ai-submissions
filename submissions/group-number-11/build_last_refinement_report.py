"""Summarize the final robustness continuation from actual saved outputs."""
import json
from pathlib import Path
import pandas as pd
T=Path(__file__).resolve().parent;O=T/'last_refinement_results';R=T/'report'
s=json.loads((O/'validate_summary.json').read_text());f=json.loads((O/'refit_summary.json').read_text()) if (O/'refit_summary.json').exists() else None
scores=pd.read_csv(O/'comparison.csv').pivot(index='model',columns='condition',values='auroc')
names={'baseline':'Original neural','new_only':'Robust continuation','equal_old_new_logit_mean':'Equal mean, 384-token views','new_only_256':'Robust, 256-token views','equal_old_new_logit_mean_256':'Equal mean, 256-token views'}
rows=[]
for name in scores.index:
 r=scores.loc[name];rows.append(names[name]+''.join(f' & {r[k]:.6f}' for k in ['clean','typo_AI_only_4percent','punctuation_removed_AI_only','lowercase_AI_only'])+r'\\')
tex=r'''\newpage
\begingroup\small\setlength{\emergencystretch}{2em}
\section*{Final robustness continuation}
The pure neural model scored 0.92433 publicly versus 0.92164 for its rank blend, reversing validation ordering. With one submission opportunity and one hour of local compute, we tested robustness continuation. Its Kaggle score is pending.

\subsection*{Changes and leakage control}
Validation starts from the saved 8,000-row model; final fitting starts from the separate model already trained on all 10,000 labels. Checkpoint and input hashes are recorded. The full-data model is never evaluated on overlapping validation labels. No external labels or prediction API are used. __COMPUTE_NOTE__

The validation model receives one additional epoch with a 384-token window, with random contiguous crops for overlength training passages. At inference, overlength passages average 384-token head and tail logits. Unicode, selected homoglyphs, quotation marks and whitespace are normalized. Training perturbations are label-independent: 50\% normalized clean, 30\% with 4\%-word typos, 10\% with random punctuation removal, and 10\% lowercase. Settings include AdamW LR $8\times10^{-6}$, effective batch 32, label smoothing 0.05, frozen embeddings, seed 2026 and local MPS bfloat16; full settings are in the reproduction guide.

\subsection*{Diagnostic comparison and adaptive window check}
Clean evaluation retains all 2,000 official validation IDs. Three synthetic conditions use the same balanced 1,000-row subset (seed 2026), altering only AI passages to simulate evasion. Their unmodified reference AUROC is 0.989270. Actual hidden attack types and prevalence are unknown. Training augmentation affects both labels; production preprocessing never uses labels.

\par\noindent
\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule
Model & Clean & AI typos & AI no punct. & AI lowercase\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabularx}\par

After the 384-token candidate lost clean AUROC, we additionally checked 256-token head/tail views on the same conditions. This window check was adaptive, not preregistered. The original rule used a clean AUROC floor of 0.985 and favored the equal neural mixture. After observing that the robust-only model wins all three stress conditions and the overall mean, we explicitly relaxed the floor to 0.98 and selected it. This trade-off revision is adaptive, not preregistered; the original rule and all results remain recorded. The condition mean is a selection heuristic, not predicted Kaggle AUROC. All candidates are neural.

'''+f"Selected: {names[s['chosen']]}. Clean validation AUROC: {s['clean_validation_auroc']:.6f}; four-condition mean: {s['selection_mean_auroc']:.6f}. Validation training and diagnostics took {s['elapsed_seconds']/60:.1f} minutes. "+(f"Final refit and inference took {f['elapsed_seconds']/60:.1f} minutes. " if f else '')+r'''

\subsection*{Reproduction and limitations}
See \texttt{FINAL\_REFINEMENT.md} and \texttt{run\_last\_refinement.py} (validate/refit modes). The separate output is \path{artifacts/submission_final_robust_neural.csv}. The verifier checks metrics, selection, IDs, transforms and hashes. Sigmoid outputs are uncalibrated.

The reused holdout and synthetic tests do not establish hidden-domain performance. Context, normalization and augmentation form a combined intervention, not isolated effects. The candidate needs actual Kaggle evaluation; private scores remain unknown.
\endgroup
'''
if f:
    compute_note=(f"A microbatch-32 attempt was discarded. The final continuation uses {f['continuation_batches']} microbatches ({f['continuation_rows_seen']:,} augmented rows; {f['continuation_epochs']:.3f} of a full-data epoch), retaining the nominal 500-microbatch learning-rate schedule. The starting model already trained on all 10,000 labels. Development validation used 500 microbatches, so its scores do not measure these deadline-limited final weights.")
else:
    compute_note="A microbatch-32 attempt was discarded. The final continuation has a deadline-limited batch budget, with exact exposure to be recorded in the final manifest. Its starting model already trained on all 10,000 labels; the development validation run used 500 microbatches."
tex=tex.replace('__COMPUTE_NOTE__',compute_note)
(R/'last_refinement_appendix.tex').write_text('\n'.join(line.rstrip() for line in tex.splitlines())+'\n')
print('Generated final refinement appendix.')
