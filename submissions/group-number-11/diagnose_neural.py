"""Post-selection training AUROC and label-blind text perturbation diagnostics.

Loads the selected validation checkpoint. Does not fit or select another model.
"""
import argparse,json,os,re,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data_dir',type=Path,required=True);p.add_argument('--dependency_dir');args=p.parse_args()
if args.dependency_dir:sys.path.append(args.dependency_dir)
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from contextlib import nullcontext
from robust_models import normalize
TEAM=Path(__file__).resolve().parent;OUT=TEAM/'neural_results'
summary=json.loads((OUT/'validate_summary.json').read_text())
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(4)
path=TEAM/'artifacts/neural/validate/best'
tokenizer=AutoTokenizer.from_pretrained(str(path),use_fast=False,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True).to(device).eval()
frame=pd.read_csv(args.data_dir/'train.csv').set_index('id')
tr=frame.loc[pd.read_csv(TEAM/'train_ids.csv').id].reset_index()
va=frame.loc[pd.read_csv(TEAM/'validation_ids.csv').id].reset_index()
def predict(texts):
    scores=[]
    with torch.inference_mode():
        for offset in range(0,len(texts),32):
            batch=tokenizer([' '.join(str(t).split()) for t in texts[offset:offset+32]],max_length=summary['max_length'],truncation=True,padding=True,pad_to_multiple_of=8,return_tensors='pt').to(device)
            context=torch.autocast(device_type=device.type,dtype=torch.bfloat16) if summary['precision']=='bf16' else nullcontext()
            with context:logits=model(**batch).logits
            scores.extend((logits[:,1]-logits[:,0]).float().cpu().numpy().tolist())
            if offset%1024==0:print('Scored',min(offset+32,len(texts)),'/',len(texts),flush=True)
    return np.array(scores)
token_rows=[]
for name,part in [('train',tr),('validation',va)]:
    lengths=np.array([len(x) for x in tokenizer([' '.join(t.split()) for t in part.text],truncation=False,padding=False)['input_ids']])
    token_rows.append({'split':name,'n':len(part),'fraction_over_256_tokens':float((lengths>256).mean()),'fraction_over_384_tokens':float((lengths>384).mean()),'median_tokens':float(np.median(lengths)),'p95_tokens':float(np.quantile(lengths,.95))})
(OUT/'token_length_summary.json').write_text(json.dumps(token_rows,indent=2)+'\n')
start=time.perf_counter()
original=predict(va.text.tolist());checkpoint_auc=roc_auc_score(va.label,original)
if not np.isclose(checkpoint_auc,summary['validation_auroc'],atol=5e-5,rtol=0):
    raise ValueError('Saved checkpoint no longer matches the selected validation model')
s=predict(tr.text.tolist())
pd.DataFrame({'id':tr.id,'label':tr.label,'decision_score':s}).to_csv(OUT/'training_predictions.csv',index=False)
results={'train_auroc':roc_auc_score(tr.label,s),'validation_auroc':summary['validation_auroc'],'reloaded_checkpoint_validation_auroc':checkpoint_auc,'best_epoch':summary['best_epoch'],'device':str(device)}
normalized=[normalize(x) for x in va.text]
for name,texts in [('format_normalized',normalized),('punctuation_removed',[re.sub(r'[^\w\s]',' ',x) for x in normalized])]:
    score=predict(texts);results[name+'_validation_auroc']=roc_auc_score(va.label,score)
    pd.DataFrame({'id':va.id,'label':va.label,'decision_score':score}).to_csv(OUT/f'{name}_scores.csv',index=False)
results['elapsed_seconds']=time.perf_counter()-start
(OUT/'diagnostics.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
