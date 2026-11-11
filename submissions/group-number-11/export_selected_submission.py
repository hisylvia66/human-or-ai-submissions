"""Export the verified selected neural predictions under the course filename."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

TEAM = Path(__file__).resolve().parent
source = TEAM / 'artifacts/submission_deberta.csv'
manifest = json.loads((TEAM / 'neural_results/refit_summary.json').read_text())
if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['prediction_sha256']:
    raise ValueError('Selected predictions do not match the recorded refit manifest.')
frame = pd.read_csv(source)
if list(frame.columns) != ['id', 'label'] or len(frame) != 3000 or not frame.id.is_unique or frame.isna().any().any() or not frame.label.between(0, 1).all():
    raise ValueError('Invalid selected submission format.')
target = TEAM / 'artifacts/final_submission/submission.csv'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, target)
print(target)
print('SHA-256:', hashlib.sha256(target.read_bytes()).hexdigest())
