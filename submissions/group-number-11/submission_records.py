"""Keep observed leaderboard scores attached to exact prediction/input hashes."""
import json
from pathlib import Path


def write_manifest(path, record):
    path=Path(path)
    previous=json.loads(path.read_text()) if path.exists() else None
    record=dict(record)
    if previous and previous.get('public_score') is not None:
        identical=(previous.get('prediction_sha256')==record.get('prediction_sha256') and
                   previous.get('input_sha256')==record.get('input_sha256'))
        if identical:
            for key in ('public_score','private_score','kaggle_status','status','public_score_evidence'):
                if key in previous:record[key]=previous[key]
        else:
            history_path=path.with_name(path.stem+'_history.json')
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            if not any(x.get('prediction_sha256')==previous.get('prediction_sha256') for x in history):
                history.append(previous)
            history_path.write_text(json.dumps(history,indent=2)+'\n')
    path.write_text(json.dumps(record,indent=2)+'\n')
