"""Build LaTeX tables/macros from committed outputs; preserve authored main.tex prose.

Run this after experiments, annotations, or report/status.json change. Refuses
unreviewed errors and inconsistent scores. Compile report/main.tex with pdfLaTeX
(Overleaf) or Tectonic. No network access is performed by this script.
"""
import json
from pathlib import Path
import shutil
import pandas as pd

TEAM = Path(__file__).resolve().parent
REPORT = TEAM/'report'
RESULTS = TEAM/'final_results'


def esc(s):
    chars = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#',
             '_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(chars.get(c,c) for c in str(s))


def table(name, headers, rows, widths=None):
    spec = widths or ('l' + 'r'*(len(headers)-1))
    lines = [r'\par\noindent',r'\begin{tabularx}{\linewidth}{@{}'+spec+r'@{}}',r'\toprule',
             ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabularx}',r'\par']
    (REPORT/'tables'/name).write_text('\n'.join(lines)+'\n')


def main():
    (REPORT/'tables').mkdir(parents=True,exist_ok=True)
    (REPORT/'figures').mkdir(exist_ok=True)
    status=json.loads((REPORT/'status.json').read_text())
    summary=json.loads((RESULTS/'evaluation_summary.json').read_text())
    selected=json.loads((RESULTS/'selected_model.json').read_text())
    if selected['config'] != 'nbsvm':
        raise ValueError('Selected model changed: update the authored report narrative before rebuilding')
    errors=pd.read_csv(RESULTS/'error_analysis.csv')
    if errors.possible_reason.str.contains('REQUIRES TEXT REVIEW').any():
        raise ValueError('Review and annotate all selected error passages first')
    if abs(summary['validation_auroc']-selected['validation_auroc'])>1e-12:
        raise ValueError('Selection and final evaluation disagree')
    macros = {
        'FinalTrainAUC':f"{summary['train_auroc']:.6f}",
        'FinalValAUC':f"{summary['validation_auroc']:.6f}",
        'FinalGap':f"{summary['train_auroc']-summary['validation_auroc']:.6f}",
        'FinalThreshold':f"{summary['classification_threshold_p_ai']:.6f}",
        'CalSlope':f"{summary['calibration_slope']:.6f}",
        'CalIntercept':f"{summary['calibration_intercept']:.6f}",
        'FinalFitTime':f"{summary['fit_seconds']:.2f}",
        'Improvement':f"{selected['delta_vs_baseline']:.6f}",
        'CILow':f"{selected['paired_bootstrap_95ci'][0]:.6f}",
        'CIHigh':f"{selected['paired_bootstrap_95ci'][1]:.6f}",
        'PublicScore':f"{status['public_score']:.5f}" if status['public_score'] is not None else 'Not yet submitted',
        'PrivateScore':f"{status['private_score']:.5f}" if status['private_score'] is not None else 'Not yet available',
        'SubmissionStatus':esc(status['submission_status']),
        'PRLink':r'\url{'+status['pr_url']+'}' if status['pr_url'] else 'PR not yet opened',
    }
    (REPORT/'metrics.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    q1=pd.read_csv(RESULTS/'q1_model_comparison.csv')
    short={'Word TF-IDF + Linear SVM':'Linear SVM ($C=1$)',
           'Word TF-IDF + Logistic Regression':'Logistic regression ($C=1$)',
           'Word TF-IDF + Complement Naive Bayes':'Complement NB ($\\alpha=1$)'}
    table('q1_comparison.tex',['Model (word TF-IDF)','Train AUC','Val. AUC','Fit (s)'],
          [[short[r.model],f'{r.train_auroc:.6f}',f'{r.validation_auroc:.6f}',f'{r.fit_time_seconds:.2f}'] for r in q1.itertuples()], 'Yrrr')
    tune=pd.read_csv(RESULTS/'q1_hyperparameter_tuning.csv')
    table('q1_tuning.tex',['Model','$C$','Train AUC','Val. AUC'],
          [[esc(r.model),f'{r.C:g}',f'{r.train_auroc:.6f}',f'{r.validation_auroc:.6f}'] for r in tune.itertuples()], 'Yrrr')
    curve=pd.read_csv(TEAM/'person_c_results/learning_curve_results.csv')
    table('q2_curve.tex',['Training rows','Train AUC','Val. AUC','Fit (s)'],
          [[f'{r.n_train:,}',f'{r.train_auroc:.6f}',f'{r.validation_auroc:.6f}',f'{r.fit_seconds:.2f}'] for r in curve.itertuples()], 'Yrrr')
    shutil.copy2(TEAM/'person_c_results/learning_curve.pdf',REPORT/'figures/learning_curve.pdf')
    q3=pd.read_csv(TEAM/'unlabeled_experiment_results.csv')
    rows=[]
    for r in q3.itertuples():
        if r.experiment=='E0':label='Supervised baseline'
        elif r.experiment=='E1':label='Fit TF-IDF on training + unlabeled'
        elif r.experiment=='E2':label=f'One round; confidence {r.threshold:g}'
        else:label=f'{r.method}; confidence {r.threshold:g}'
        rows.append([esc(label),int(r.n_pseudo),f'{r.val_auroc:.6f}',f'{r.runtime_s:.2f}'])
    table('q3_results.tex',['Method','Pseudo rows','Val. AUC','Time (s)'],rows,'Yrrr')
    groups=pd.read_csv(RESULTS/'length_group_results.csv')
    table('q4_groups.tex',['Group (words)','Count','Human / AI','Val. AUC','FP / FN'],
          [[f'{r.length_group.title()} ({r.min_words}--{r.max_words})',r.n,f'{r.n_human} / {r.n_ai}',f'{r.validation_auroc:.6f}',f'{r.false_positives} / {r.false_negatives}'] for r in groups.itertuples()],'Yrrrr')
    cm=summary['confusion_matrix']
    table('q4_confusion.tex',['True class','Predicted human','Predicted AI'],[['Human',cm[0][0],cm[0][1]],['AI',cm[1][0],cm[1][1]]],'Yrr')
    notes=json.loads((TEAM/'final_error_notes.json').read_text())
    table('q4_errors.tex',['ID','Error','P(AI)','Observed features / possible failure'],
          [[r.id,r.error_type,f'{r.p_ai:.6f}',esc(notes[str(r.id)]['report_reason'])] for r in errors.itertuples()], 'rrrY')
    q5=pd.read_csv(TEAM/'robustness_improvement_results.csv')
    table('q5_robustness.tex',['Configuration','Original','Typos','Homoglyph','Zero-width','Fit (s)'],
          [[r.config,f'{r.val_clean:.4f}',f'{r.val_typo:.4f}',f'{r.val_homoglyph:.4f}',f'{r.val_zero_width:.4f}',f'{r.fit_s:.2f}'] for r in q5.itertuples()],'Yrrrrr')
    refine=pd.read_csv(RESULTS/'refinement_results.csv')
    names={'baseline':'Original word + character SVM','char_weight2':'Character block weighted 2x','char_only':'Character-only SVM','logreg':'Word + character logistic regression',
           'nbsvm':'NB-weighted word + character SVM','selftrain2':'Self-training, two rounds (revised)'}
    table('refinement.tex',['Candidate','Train AUC','Val. AUC','Fit (s)'],
          [[esc(names[r.config]),f'{r.train_auroc:.6f}',f'{r.validation_auroc:.6f}',f'{r.fit_seconds:.2f}'] for r in refine.itertuples()],'Yrrr')
    print('LaTeX tables, metrics and figure generated from verified outputs.')


if __name__=='__main__':
    main()
