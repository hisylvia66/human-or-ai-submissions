"""Download a pinned, general-purpose pretrained backbone for local training only."""
import argparse,json,os,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--dependency_dir');p.add_argument('--model',default='microsoft/deberta-v3-small')
p.add_argument('--revision',default='a36c739020e01763fe789b4b85e2df55d6180012');a=p.parse_args()
if a.dependency_dir:sys.path.append(a.dependency_dir)
TEAM=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,snapshot_download
info=HfApi().model_info(a.model,revision=a.revision)
path=snapshot_download(a.model,revision=info.sha,cache_dir=str(TEAM/'artifacts/hf_cache'),
                       allow_patterns=['config.json','tokenizer*','special_tokens_map.json','spm.model','*.safetensors','pytorch_model.bin'],max_workers=2)
(TEAM/'neural_results').mkdir(exist_ok=True)
record=dict(model_id=a.model,revision=info.sha,snapshot_dir=str(Path(path).relative_to(TEAM)),
            model_purpose='General language pretraining, not an externally supervised AI-text detector',
            supervised_labels='Only supplied competition train.csv',external_prediction_api_used=False)
(TEAM/'neural_results/pretrained_source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
