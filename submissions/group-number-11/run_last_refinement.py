"""Time-bounded local robustness continuation with fixed-ID validation.

Uses separate 8k and 10k starting checkpoints; never validates a model trained on
validation labels. No external labels or test-label access. Outputs stay separate.
"""
import argparse,hashlib,json,math,os,random,re,sys,time,unicodedata
from pathlib import Path
from contextlib import nullcontext
p=argparse.ArgumentParser();p.add_argument('--data_dir',type=Path,required=True);p.add_argument('--dependency_dir');p.add_argument('--mode',choices=['validate','refit','diagnose','assess_views'],required=True);p.add_argument('--device',choices=['auto','mps','cuda','cpu'],default='auto');p.add_argument('--batch_size',type=int,default=16);p.add_argument('--accumulation',type=int,default=2);p.add_argument('--max_batches',type=int,default=0);p.add_argument('--schedule_batches',type=int,default=0);p.add_argument('--prediction_batch_size',type=int,default=24);a=p.parse_args()
assert a.batch_size*a.accumulation==32, 'Keep effective optimizer batch size fixed at 32'
if a.dependency_dir:sys.path.append(a.dependency_dir)
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from robust_models import normalize
from submission_records import write_manifest

T=Path(__file__).resolve().parent;O=T/'last_refinement_results';O.mkdir(exist_ok=True)
A=T/'artifacts/last_refinement';A.mkdir(exist_ok=True)
SEED=2026;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
device_name=('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) if a.device=='auto' else a.device
D=torch.device(device_name)
use_bf16=D.type=='mps' or (D.type=='cuda' and torch.cuda.is_bf16_supported())
def mixed_precision():
    return torch.autocast(device_type=D.type,dtype=torch.bfloat16) if use_bf16 else nullcontext()
full=pd.read_csv(a.data_dir/'train.csv');idx=full.set_index('id')
tr=idx.loc[pd.read_csv(T/'train_ids.csv').id].reset_index();va=idx.loc[pd.read_csv(T/'validation_ids.csv').id].reset_index()
assert len(tr)==8000 and len(va)==2000 and not set(tr.id)&set(va.id)
start_checkpoint=(A/'validate') if a.mode in ['diagnose','assess_views'] else T/'artifacts/neural'/('validate/best' if a.mode=='validate' else 'refit/final')
tokenizer=AutoTokenizer.from_pretrained(str(start_checkpoint),local_files_only=True,use_fast=False)
def load_model(path):return AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True).to(D)
model=load_model(start_checkpoint)
for parameter in model.deberta.embeddings.parameters():parameter.requires_grad=False

def typos(text,rng):
    words=text.split()
    for j,w in enumerate(words):
        if len(w)>3 and rng.random()<.04:
            k=rng.randrange(len(w)-1)
            words[j]=w[:k]+w[k+1]+w[k]+w[k+2:] if rng.random()<.5 else w[:k]+w[k+1:]
    return ' '.join(words)

def perturb(text,rng):
    text=normalize(text);r=rng.random()
    if r<.30:return typos(text,rng)
    if r<.40:return ''.join(' ' if unicodedata.category(c).startswith('P') and rng.random()<.5 else c for c in text)
    if r<.50:return text.lower()
    return text

def encode(text,limit=384,tail=False,augment=False,seed=SEED):
    rng=random.Random(seed)
    text=perturb(text,rng) if augment else normalize(text)
    tokens=tokenizer.encode(text,add_special_tokens=False)
    n=limit-2
    if len(tokens)>n:
        begin=rng.randint(0,len(tokens)-n) if augment else (len(tokens)-n if tail else 0)
        tokens=tokens[begin:begin+n]
    ids=tokenizer.build_inputs_with_special_tokens(tokens)
    return {'input_ids':ids,'attention_mask':[1]*len(ids)}
class Encoded(Dataset):
    def __init__(self,frame,augment=False,limit=384,tail=False):
        self.items=[encode(text,limit,tail,augment,SEED+int(i)) for i,text in zip(frame.id,frame.text)]
        self.labels=frame.label.to_numpy() if augment else None
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        x=dict(self.items[i])
        if self.labels is not None:x['labels']=int(self.labels[i])
        return x

def collate(items):return tokenizer.pad(items,padding=True,pad_to_multiple_of=8,return_tensors='pt')
def infer(frame,limit=384,views=True,raw_baseline=False):
    model.eval()
    if raw_baseline:
        texts=[' '.join(t.split()) for t in frame.text]
        encoded=tokenizer(texts,truncation=True,max_length=256,padding=False)
        items=[{k:v[i] for k,v in encoded.items()} for i in range(len(frame))]
    else:items=Encoded(frame,limit=limit)
    scores=[]
    with torch.inference_mode():
        for batch in DataLoader(items,batch_size=a.prediction_batch_size,collate_fn=collate,num_workers=0):
            batch={k:v.to(D) for k,v in batch.items()}
            with mixed_precision():z=model(**batch).logits
            scores.extend((z[:,1]-z[:,0]).float().cpu().numpy())
    scores=np.asarray(scores,dtype=float)
    if views and not raw_baseline:
        long=np.array([len(tokenizer.encode(normalize(t),add_special_tokens=False))>limit-2 for t in frame.text])
        if long.any():
            tail_scores=[]
            with torch.inference_mode():
                for batch in DataLoader(Encoded(frame.loc[long],limit=limit,tail=True),batch_size=a.prediction_batch_size,collate_fn=collate,num_workers=0):
                    batch={k:v.to(D) for k,v in batch.items()}
                    with mixed_precision():z=model(**batch).logits
                    tail_scores.extend((z[:,1]-z[:,0]).float().cpu().numpy())
            scores[long]=.5*(scores[long]+np.asarray(tail_scores))
    assert np.isfinite(scores).all()
    return scores

# Stress conditions share a fixed, balanced 1,000-row subset of the official holdout.
stress=pd.concat([va[va.label==c].sample(n=500,random_state=SEED) for c in [0,1]]).reset_index(drop=True)
conditions={'clean':va}
for name in ['typo_AI_only_4percent','punctuation_removed_AI_only','lowercase_AI_only']:
    f=stress.copy();texts=[]
    for row in f.itertuples():
        t=row.text
        if row.label==1:
            if name.startswith('typo'):t=typos(' '.join(t.split()),random.Random(SEED+int(row.id)))
            elif name.startswith('punctuation'):t=re.sub(r'[^\w\s]',' ',t)
            else:t=t.lower()
        texts.append(t)
    f['text']=texts;conditions[name]=f

start=time.perf_counter()
if a.mode=='assess_views':
    summary=json.loads((O/'validate_summary.json').read_text())
    results=pd.read_csv(O/'comparison.csv').to_dict('records')
    for name,frame in conditions.items():
        new=infer(frame,limit=256);old=pd.read_csv(O/f'baseline_{name}.csv').set_index('id').loc[frame.id,'score'].to_numpy()
        pd.DataFrame({'id':frame.id,'label':frame.label,'new_score':new,'old_score':old}).to_csv(O/f'validation_256_{name}.csv',index=False)
        for kind,score in [('new_only_256',new),('equal_old_new_logit_mean_256',.5*(new+old))]:
            results.append({'model':kind,'condition':name,'n':len(frame),'auroc':roc_auc_score(frame.label,score)})
        print('Additional 256-token views',name,results[-2:],flush=True)
    table=pd.DataFrame(results);table.to_csv(O/'comparison.csv',index=False)
    wide=table.pivot(index='model',columns='condition',values='auroc');eligible=wide[wide.clean>=.985]
    chosen=eligible.mean(axis=1).idxmax() if len(eligible) else wide.clean.idxmax()
    limit=256 if chosen.endswith('_256') else 384
    summary.update(chosen=chosen,inference_limit=limit,inference=f'normalized {limit}-token head/tail mean for overlength inputs',selection_mean_auroc=float(wide.loc[chosen].mean()),clean_validation_auroc=float(wide.loc[chosen,'clean']),additional_view_seconds=time.perf_counter()-start)
    (O/'validate_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True);sys.exit(0)
if a.mode=='diagnose':
    selection=json.loads((O/'validate_summary.json').read_text());chosen=selection['chosen']
    new=infer(tr,limit=selection.get('inference_limit',384))
    old=pd.read_csv(T/'neural_results/training_predictions.csv').set_index('id').loc[tr.id,'decision_score'].to_numpy()
    score=new if chosen.startswith('new_only') else (.5*(new+old) if chosen.startswith('equal_old_new_logit_mean') else old)
    pd.DataFrame({'id':tr.id,'label':tr.label,'new_score':new,'old_score':old,'selected_score':score}).to_csv(O/'training_predictions.csv',index=False)
    record={'n_train':len(tr),'chosen':chosen,'train_auroc':roc_auc_score(tr.label,score),'new_model_train_auroc':roc_auc_score(tr.label,new),'elapsed_seconds':time.perf_counter()-start}
    (O/'training_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True);sys.exit(0)
if a.mode=='validate':
    prior=pd.read_csv(T/'neural_results/validation_predictions.csv').set_index('id')
    for name,frame in conditions.items():
        scores=prior.loc[frame.id,'decision_score'].to_numpy() if name=='clean' else infer(frame,views=False,raw_baseline=True)
        pd.DataFrame({'id':frame.id,'label':frame.label,'score':scores}).to_csv(O/f'baseline_{name}.csv',index=False)
        print('Baseline',name,roc_auc_score(frame.label,scores),'elapsed',round(time.perf_counter()-start,1),flush=True)

fit=tr if a.mode=='validate' else full
loader=DataLoader(Encoded(fit,augment=True),batch_size=a.batch_size,shuffle=True,collate_fn=collate,num_workers=0,generator=torch.Generator().manual_seed(SEED))
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-6,weight_decay=.01)
effective_batches=min(len(loader),a.max_batches) if a.max_batches else len(loader)
schedule_batches=a.schedule_batches or effective_batches
updates=math.ceil(schedule_batches/a.accumulation);scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(.05*updates)),updates)
model.train();optimizer.zero_grad(set_to_none=True);losses=[];last=time.perf_counter()
for step,batch in enumerate(loader,1):
    labels=batch.pop('labels').to(D);batch={k:v.to(D) for k,v in batch.items()}
    with mixed_precision():
        logits=model(**batch).logits
        loss=torch.nn.functional.cross_entropy(logits,labels,label_smoothing=.05)
    if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
    remainder=effective_batches%a.accumulation
    divisor=remainder if remainder and step>effective_batches-remainder else a.accumulation
    (loss/divisor).backward();losses.append(float(loss.detach().cpu()))
    if step%a.accumulation==0 or step==effective_batches:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    if step==1 or time.perf_counter()-last>=25:
        print(json.dumps({'mode':a.mode,'batch':step,'total_batches':effective_batches,'loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-start}),flush=True);last=time.perf_counter()
    if step>=effective_batches:break
checkpoint=A/a.mode;model.save_pretrained(checkpoint,safe_serialization=True);tokenizer.save_pretrained(checkpoint)
record={'mode':a.mode,'device':str(D),'precision':'bf16' if use_bf16 else 'fp32','torch_version':torch.__version__,'microbatch_size':a.batch_size,'accumulation':a.accumulation,'effective_batch_size':32,'n_train':len(fit),'seed':SEED,'continuation_epochs':effective_batches/len(loader),'continuation_rows_seen':min(len(fit),effective_batches*a.batch_size),'continuation_batches':effective_batches,'schedule_batches':schedule_batches,'prediction_batch_size':a.prediction_batch_size,'learning_rate':8e-6,'label_smoothing':.05,'max_length':384,'inference':'normalized 384-token head/tail mean for overlength inputs','starting_checkpoint':str(start_checkpoint.relative_to(T)),'train_loss':float(np.mean(losses)),'training_elapsed_seconds':time.perf_counter()-start,'input_sha256':{n:hashlib.sha256((a.data_dir/n).read_bytes()).hexdigest() for n in ['train.csv','test.csv','sample_submission.csv']}}
if a.mode=='validate':
    results=[]
    for name,frame in conditions.items():
        new=infer(frame);old=pd.read_csv(O/f'baseline_{name}.csv').set_index('id').loc[frame.id,'score'].to_numpy()
        pd.DataFrame({'id':frame.id,'label':frame.label,'new_score':new,'old_score':old}).to_csv(O/f'validation_{name}.csv',index=False)
        for kind,s in [('baseline',old),('new_only',new),('equal_old_new_logit_mean',.5*(new+old))]:
            results.append({'model':kind,'condition':name,'n':len(frame),'auroc':roc_auc_score(frame.label,s)})
        print('Evaluated',name,results[-3:],'elapsed',round(time.perf_counter()-start,1),flush=True)
    table=pd.DataFrame(results);table.to_csv(O/'comparison.csv',index=False)
    wide=table.pivot(index='model',columns='condition',values='auroc');eligible=wide[wide.clean>=.985]
    chosen=eligible.mean(axis=1).idxmax() if len(eligible) else wide.clean.idxmax()
    record.update(chosen=chosen,selection_mean_auroc=float(wide.loc[chosen].mean()),clean_validation_auroc=float(wide.loc[chosen,'clean']))
else:
    selection=json.loads((O/'validate_summary.json').read_text());chosen=selection['chosen']
    test=pd.read_csv(a.data_dir/'test.csv');sample=pd.read_csv(a.data_dir/'sample_submission.csv');limit=selection.get('inference_limit',384);new=infer(test,limit=limit)
    record['inference']=selection['inference'];record['inference_limit']=limit
    old=pd.read_csv(T/'artifacts/deberta_test_scores.csv').set_index('id').loc[test.id,'decision_score'].to_numpy()
    score=new if chosen.startswith('new_only') else (.5*(new+old) if chosen.startswith('equal_old_new_logit_mean') else old)
    pd.DataFrame({'id':test.id,'new_score':new,'old_score':old,'selected_score':score}).to_csv(A/'test_scores.csv',index=False)
    pred=pd.DataFrame({'id':test.id,'label':expit(score)})
    sub=sample[['id']].merge(pred,on='id',how='left',validate='one_to_one')
    assert len(sub)==3000 and sub.notna().all().all() and sub.label.between(0,1).all()
    path=T/'artifacts/submission_final_robust_neural.csv';sub.to_csv(path,index=False)
    record.update(chosen=chosen,prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),public_score=None,private_score=None)
    print('Saved',path,flush=True)
record['elapsed_seconds']=time.perf_counter()-start
write_manifest(O/f'{a.mode}_summary.json',record)
print(json.dumps(record,indent=2),flush=True)
