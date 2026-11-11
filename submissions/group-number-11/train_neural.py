"""Local supervised DeBERTa experiment. Uses supplied labels and committed IDs only.

Modes: benchmark (short timing run), validate (8k/2k), topic (training-only grouped
holdout), refit (all 10k, chosen epoch count, then local test inference).
Weights/data remain under ignored artifacts/data directories.
"""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--data_dir',type=Path,required=True)
parser.add_argument('--dependency_dir')
parser.add_argument('--mode',choices=['benchmark','validate','topic','refit'],default='validate')
parser.add_argument('--epochs',type=int,default=3)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--accumulation',type=int,default=2)
parser.add_argument('--max_length',type=int,default=384)
parser.add_argument('--learning_rate',type=float,default=2e-5)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--max_steps',type=int,default=12)
parser.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
parser.add_argument('--freeze_embeddings',action='store_true')
args=parser.parse_args()
if args.dependency_dir:sys.path.append(args.dependency_dir)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup

TEAM=Path(__file__).resolve().parent
OUT=TEAM/'neural_results';OUT.mkdir(exist_ok=True)
if args.mode=='refit':
    chosen=json.loads((OUT/'validate_summary.json').read_text())
    for key in ('learning_rate','batch_size','accumulation','max_length','seed','precision','freeze_embeddings'):
        setattr(args,key,chosen[key])
    args.epochs=chosen['best_epoch']
CHECKPOINT=TEAM/'artifacts/neural'/args.mode;CHECKPOINT.mkdir(parents=True,exist_ok=True)
random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
torch.set_num_threads(4)
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
source=json.loads((OUT/'pretrained_source.json').read_text())
source_path=Path(source.get('snapshot_dir',source.get('local_path')))
if not source_path.is_absolute():source_path=TEAM/source_path
print('Device:',device,'pretrained revision:',source['revision'],flush=True)
tokenizer=AutoTokenizer.from_pretrained(str(source_path),use_fast=False,local_files_only=True,trust_remote_code=False)
model=AutoModelForSequenceClassification.from_pretrained(str(source_path),num_labels=2,local_files_only=True,trust_remote_code=False)
if args.freeze_embeddings:
    for parameter in model.deberta.embeddings.parameters():parameter.requires_grad=False
model.to(device)
from contextlib import nullcontext
def precision_context():
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16) if args.precision=='bf16' else nullcontext()

class TextData(Dataset):
    def __init__(self,frame):
        self.ids=frame.id.to_numpy()
        # Uniform whitespace removes a known formatting shortcut; no labels used.
        self.encoded=tokenizer([' '.join(t.split()) for t in frame.text],truncation=True,max_length=args.max_length,padding=False)
        self.labels=frame.label.to_numpy() if 'label' in frame else None
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        item={k:v[i] for k,v in self.encoded.items()}
        if self.labels is not None:item['labels']=int(self.labels[i])
        return item

def collate(items):return tokenizer.pad(items,padding=True,pad_to_multiple_of=8,return_tensors='pt')

def predict(dataset):
    model.eval();scores=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=args.batch_size*2,shuffle=False,collate_fn=collate,num_workers=0):
            batch={k:v.to(device) for k,v in batch.items() if k!='labels'}
            with precision_context():logits=model(**batch).logits
            scores.extend((logits[:,1]-logits[:,0]).float().cpu().numpy().tolist())
    return np.asarray(scores)

full=pd.read_csv(args.data_dir/'train.csv')
trids=pd.read_csv(TEAM/'train_ids.csv').id;vaids=pd.read_csv(TEAM/'validation_ids.csv').id
assert len(full)==10000 and full.id.is_unique and not set(trids)&set(vaids)
assert set(trids)|set(vaids)==set(full.id)
tr=full.set_index('id').loc[trids].reset_index();va=full.set_index('id').loc[vaids].reset_index()
if args.mode=='topic':
    groups=pd.read_csv(TEAM/'audit_results/training_topic_clusters.csv').set_index('id').loc[tr.id].topic_cluster
    fit,held=next(GroupKFold(3).split(tr,tr.label,groups))
    tr,va=tr.iloc[fit].reset_index(drop=True),tr.iloc[held].reset_index(drop=True)
elif args.mode=='refit':
    tr=full
    # Hyperparameters/epoch count selected by validation; no new test feedback.
    chosen=json.loads((OUT/'validate_summary.json').read_text())
    args.epochs=chosen['best_epoch']
train_data=TextData(tr);val_data=TextData(va) if args.mode in ['validate','topic'] else None
loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,collate_fn=collate,num_workers=0,
                  generator=torch.Generator().manual_seed(args.seed))
optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
steps_per_epoch=math.ceil(len(loader)/args.accumulation)
scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(.1*steps_per_epoch*args.epochs)),steps_per_epoch*args.epochs)
metrics=[];best=-1.;best_epoch=0;start=time.perf_counter();steps=0;last_log=start
print('Rows:',len(tr),'steps/epoch:',len(loader),'epochs:',args.epochs,flush=True)
for epoch in range(1,args.epochs+1):
    model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0
    for step,batch in enumerate(loader,1):
        batch={k:v.to(device) for k,v in batch.items()}
        with precision_context():
            outputs=model(**batch);loss=outputs.loss
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        (loss/args.accumulation).backward();loss_sum+=float(loss.detach().cpu())
        if step%args.accumulation==0 or step==len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        steps+=1
        now=time.perf_counter()
        if step==1 or now-last_log>=25:
            print(json.dumps({'epoch':epoch,'batch':step,'batches_per_epoch':len(loader),'mean_loss':loss_sum/step,'elapsed_seconds':now-start}),flush=True);last_log=now
        if args.mode=='benchmark' and steps>=args.max_steps:
            torch.mps.synchronize() if device.type=='mps' else None
            result={'device':str(device),'steps':steps,'elapsed_seconds':time.perf_counter()-start,
                    'batch_size':args.batch_size,'max_length':args.max_length,'precision':args.precision,'freeze_embeddings':args.freeze_embeddings}
            (OUT/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True);sys.exit(0)
    if val_data is not None:
        s=predict(val_data);auc=roc_auc_score(va.label,s)
        row={'epoch':epoch,'train_loss':loss_sum/len(loader),'validation_auroc':auc,'elapsed_seconds':time.perf_counter()-start}
        metrics.append(row);pd.DataFrame(metrics).to_csv(OUT/f'{args.mode}_epochs.csv',index=False)
        pd.DataFrame({'id':va.id,'label':va.label,'decision_score':s}).to_csv(OUT/f'{args.mode}_epoch_{epoch}_scores.csv',index=False)
        print(json.dumps(row),flush=True)
        if auc>best:
            best=auc;best_epoch=epoch
            model.save_pretrained(CHECKPOINT/'best',safe_serialization=True)
            tokenizer.save_pretrained(CHECKPOINT/'best')
    else:
        print(json.dumps({'epoch':epoch,'train_loss':loss_sum/len(loader),'elapsed_seconds':time.perf_counter()-start}),flush=True)

record={'model_id':source['model_id'],'revision':source['revision'],'mode':args.mode,'device':str(device),
        'n_train':len(tr),'n_validation':len(va) if val_data is not None else 0,
        'best_epoch':best_epoch if val_data is not None else args.epochs,
        'validation_auroc':best if val_data is not None else None,
        'learning_rate':args.learning_rate,'batch_size':args.batch_size,'accumulation':args.accumulation,
        'max_length':args.max_length,'precision':args.precision,'freeze_embeddings':args.freeze_embeddings,'seed':args.seed,'whitespace_normalization':True,
        'elapsed_seconds':time.perf_counter()-start,'torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'train_csv_sha256':hashlib.sha256((args.data_dir/'train.csv').read_bytes()).hexdigest()}
record['split_sha256']={name:hashlib.sha256((TEAM/name).read_bytes()).hexdigest()
                        for name in ('train_ids.csv','validation_ids.csv')}
if args.mode=='refit':
    model.save_pretrained(CHECKPOINT/'final',safe_serialization=True);tokenizer.save_pretrained(CHECKPOINT/'final')
    test=pd.read_csv(args.data_dir/'test.csv');sample=pd.read_csv(args.data_dir/'sample_submission.csv')
    score=predict(TextData(test))
    from scipy.special import expit
    pd.DataFrame({'id':test.id,'decision_score':score}).to_csv(TEAM/'artifacts/deberta_test_scores.csv',index=False)
    record['test_scores_sha256']=hashlib.sha256((TEAM/'artifacts/deberta_test_scores.csv').read_bytes()).hexdigest()
    pred=pd.DataFrame({'id':test.id,'label':expit(score)})
    sub=sample[['id']].merge(pred,on='id',how='left',validate='one_to_one')
    assert len(sub)==3000 and sub.notna().all().all() and sub.label.between(0,1).all()
    path=TEAM/'artifacts/submission_deberta.csv';sub.to_csv(path,index=False)
    record.update(prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),public_score=None,private_score=None)
    record['test_csv_sha256']=hashlib.sha256((args.data_dir/'test.csv').read_bytes()).hexdigest()
    record['sample_submission_sha256']=hashlib.sha256((args.data_dir/'sample_submission.csv').read_bytes()).hexdigest()
    record['input_sha256']={name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                            for name in ('train.csv','test.csv','sample_submission.csv')}
    record['elapsed_seconds']=time.perf_counter()-start
    print('Saved',path,flush=True)
from submission_records import write_manifest
write_manifest(OUT/f'{args.mode}_summary.json',record)
print(json.dumps(record,indent=2),flush=True)
