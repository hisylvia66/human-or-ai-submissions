"""Generate a separately identified local-neural experiment appendix."""
import json
from pathlib import Path
import pandas as pd
T=Path(__file__).resolve().parent;O=T/'neural_results';R=T/'report'
s=json.loads((O/'validate_summary.json').read_text())
e=json.loads((O/'evaluation.json').read_text())
c=pd.read_csv(O/'candidate_comparison.csv')
rows='\n'.join(name.replace('_',r'\_')+f' & {auc:.6f} '+r'\\' for name,auc in zip(c.model,c.validation_auroc))
epochs=pd.read_csv(O/'validate_epochs.csv')
epoch_text='; '.join(f"epoch {int(x.epoch)}: {x.validation_auroc:.6f}" for x in epochs.itertuples())
diagnostics=''
if (O/'diagnostics.json').exists():
    d=json.loads((O/'diagnostics.json').read_text())
    diagnostics=(f"Selected-checkpoint training AUROC is {d['train_auroc']:.6f}; validation AUROC is {d['validation_auroc']:.6f}. "
                 f"Normalizing Unicode, quotation marks and whitespace in both classes gives {d['format_normalized_validation_auroc']:.6f}; "
                 f"also removing punctuation gives {d['punctuation_removed_validation_auroc']:.6f}. These are synthetic stress tests, not measured hidden-test conditions.")
tex=r'''\newpage
\begingroup\small\setlength{\emergencystretch}{2em}
\section*{Final DeBERTa model: method and measured results}
\textbf{Status.} The pure neural model scored 0.92433 publicly; the fixed neural/NB-style rank blend scored 0.92164. Both improved on the earlier NB/style blend (0.88628), but the locally better rank blend was worse publicly. Private scores remain unknown.

\subsection*{Hypothesis and experimental protocol}
After observing the validation/public gap, we tested whether contextual pretrained representations could complement local n-grams and stylistic statistics. We used Microsoft's general-purpose \href{https://huggingface.co/microsoft/deberta-v3-small}{DeBERTa-v3-small}, following \href{https://arxiv.org/abs/2111.09543}{DeBERTaV3 (He et al.)}. The exact weight revision is recorded in \texttt{neural\_results/pretrained\_source.json}. Pretraining is external general-language training; the only supervised labels in our fine-tuning are the supplied competition labels. All training and prediction run locally, with no external inference API or hidden labels.

Training uses the same 8,000 training IDs and evaluates the same 2,000 validation IDs. We normalize whitespace, truncate to 256 tokens, freeze pretrained embeddings, and fine-tune the encoder and new classification head. Settings: AdamW, learning rate $3\times10^{-5}$, weight decay 0.01, batch 16, accumulation 2, 10\% linear warmup followed by linear decay, gradient clipping at 1, seed 42, bfloat16 on Apple MPS, two epochs. We retain the better validation checkpoint. Validation labels select the epoch but never enter gradient updates. Test labels are unavailable.

\subsection*{Measured local results}
'''+epoch_text+f". Selected epoch: {s['best_epoch']}. Validation-run time (including evaluation and checkpointing): {s['elapsed_seconds']/60:.1f} minutes.\n\n"+r'''
\par\noindent
\begin{tabularx}{\linewidth}{@{}Yr@{}}\toprule
Model & Fixed-validation AUROC\\\midrule
'''+rows+r'''
\bottomrule\end{tabularx}\par

We also test one fixed 50:50 mean-percentile-rank blend of the neural scores and the previously submitted NB/style scores; there is no weight sweep. Ranks are computed separately within validation and test predictions without labels. These ensemble outputs are ranking scores, not calibrated probabilities. Checkpoint and candidate selection reuse the holdout; all comparisons are exploratory.

'''+diagnostics+r'''

\subsection*{Limitations and submission procedure}
About 50.5\% of validation texts exceed 256 tokens, so truncation discards potentially useful evidence. A 384-token limit would truncate 8.4\%, but the later combined robustness experiment changes several factors and does not isolate context length. Neural softmax outputs are uncalibrated. Epoch selection and repeated reuse of validation may inflate apparent gains. The observed public scores remain below validation scores, showing that the distribution gap persists.

The final neural candidate is retrained from the pinned pretrained weights on all 10,000 supplied labeled rows using the chosen epoch count and training settings. Its file is \path{artifacts/submission_deberta.csv}; the fixed ensemble is \path{artifacts/submission_deberta_robust_rank50.csv}. Prediction hashes and input hashes are recorded separately, so earlier leaderboard scores cannot transfer to new predictions. See \texttt{NEURAL\_EXPERIMENT.md} for reproduction and limitations. The later robustness continuation scored 0.91085 and was rejected; the original pure model remains the final selection.
\endgroup
'''
groups=pd.read_csv(O/'length_group_comparison.csv')
group_rows='\n'.join(f'{r.length_group.title()} & {r.n} & {r.validation_auroc:.6f} & {r.submitted_blend_auroc:.6f} '+r'\\' for r in groups.itertuples())
errors=pd.read_csv(O/'error_analysis.csv')
if errors.possible_reason.str.contains('REQUIRES TEXT REVIEW').any():
    raise ValueError('Inspect new neural errors and update neural_error_notes.json first')
from build_report import esc
error_rows='\n'.join(f'{r.id} & {r.error_type} & {r.p_ai_uncalibrated:.6f} & '+esc(r.possible_reason)+r'\\' for r in errors.itertuples())
tex+=r'''\newpage
\begingroup\small\setlength{\emergencystretch}{2em}
\section*{Neural candidate: subgroup and error diagnostics}
The same training-derived word-count boundaries (175 and 240 words) define short, medium and long validation groups. AUROC evaluates ranking; the neural classifier's default logit-zero rule corresponds to an uncalibrated softmax value of 0.5.
\par\noindent
\begin{tabularx}{\linewidth}{@{}Yrrr@{}}\toprule
Length group & Count & Neural AUROC & Submitted blend\\\midrule
'''+group_rows+r'''
\bottomrule\end{tabularx}\par
At the default threshold, the neural model produces 204 false positives and 18 false negatives (88.9\% accuracy). Its high AUROC does not imply a good operating threshold or calibrated probabilities. No validation-optimized threshold is presented as an unbiased test result.

\subsection*{Five false positives and five false negatives}
These are the most confident errors of each type at the default threshold. They are selected examples, not a representative prevalence estimate. Passage inspection supports the following hypotheses; it does not establish causal feature attribution or that supplied labels are incorrect.
\par\noindent
\begin{tabularx}{\linewidth}{@{}rrrY@{}}\toprule
ID & Type & Softmax AI & Possible reason from inspected passage\\\midrule
'''+error_rows+r'''
\bottomrule\end{tabularx}\par
Human research abstracts and structured explanations account for several confident false positives. Recipes, informal reviews and detailed biographies still produce false negatives. These genres occur in both classes. A next controlled experiment could compare 256 versus 384 tokens while retaining the same training/validation IDs; this would test whether later evidence helps rather than assuming truncation caused these errors. More representative held-out domains and probability calibration deserve separate evaluation.
\endgroup
'''
refit_path=O/'refit_summary.json'
if refit_path.exists():
    refit=json.loads(refit_path.read_text())
    tex=tex.replace('The later robustness continuation scored 0.91085 and was rejected; the original pure model remains the final selection.', f"Full-data refit and test inference took {refit['elapsed_seconds']/60:.1f} minutes. The later robustness continuation scored 0.91085 and was rejected; the original pure model remains the final selection.")
cm=e['confusion_matrix_at_logit_zero']
tex=tex.replace('204 false positives and 18 false negatives (88.9\\% accuracy)', f'{cm[0][1]} false positives and {cm[1][0]} false negatives ({100*(cm[0][0]+cm[1][1])/sum(map(sum,cm)):.1f}\\% accuracy)')
(R/'neural_appendix.tex').write_text(tex)
print('Generated neural appendix from saved results.')

method, q4 = tex.split(r'\section*{Neural candidate: subgroup and error diagnostics}', 1)
method = method.rsplit(r'\newpage', 1)[0]
q4 = r'\newpage\begingroup\small\setlength{\emergencystretch}{2em}' + '\n' + r'\section*{Q4. Generalization and error analysis: final DeBERTa}' + '\n' + r'\subsection*{Q4.1. Training versus validation}' + '\n' + f"Training AUROC is {d['train_auroc']:.6f}; fixed-validation AUROC is {d['validation_auroc']:.6f}, a gap of {d['train_auroc']-d['validation_auroc']:.6f}. The gap is consistent with some overfitting; repeated selection also limits the estimate. Public AUROC falls to 0.92433, showing a larger distribution/generalization gap. These diagnostics use the development checkpoint, not the all-label final fit.\n" + r'\subsection*{Q4.2. Text-length subgroups}' + '\n' + q4
q4=q4.replace('Five false positives and five false negatives','Q4.3. Five false positives and five false negatives').replace('Human research abstracts and structured explanations',r'\subsection*{Q4.4. Failure patterns and targeted response}'+'\n'+'Human research abstracts and structured explanations')
q4=q4.replace('A next controlled experiment could compare 256 versus 384 tokens while retaining the same training/validation IDs; this would test whether later evidence helps rather than assuming truncation caused these errors.', 'A targeted follow-up would compare only 256 versus 384 tokens with all other settings fixed, testing whether later evidence helps. The later robustness continuation changes several factors and cannot isolate this hypothesis.')
(R/'neural_method.tex').write_text(method)
q4=q4.replace('These are the most confident errors of each type at the default threshold. They are selected examples, not a representative prevalence estimate. Passage inspection supports the following hypotheses; it does not establish causal feature attribution or that supplied labels are incorrect.', 'FP denotes human text predicted AI; FN denotes AI text predicted human. These are five confident errors of each type, not a representative sample. Reasons are inspection-based hypotheses, not proven causes or label corrections.')
start=q4.index('Human research abstracts and structured explanations')
q4=q4[:start]+r'Human abstracts and structured explanations cause confident FP; recipes, informal reviews and biographies cause FN. Shared genres obscure authorship. A controlled 256-versus-384-token comparison with other settings fixed could test whether later context helps. The robustness continuation changes several factors and cannot isolate that hypothesis. Separate domains and calibration need further evaluation.'+'\n'+r'\endgroup'+'\n'
(R/'neural_q4.tex').write_text(q4)
