"""Tiny append-only JSONL experiment tracker and model-card writer."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path


def log_experiment(name: str, metrics: dict, params: dict | None = None, tags: dict | None = None, path: str = 'experiments/runs.jsonl') -> dict:
    if not name.strip(): raise ValueError('Experiment name cannot be empty')
    if not isinstance(metrics, dict) or not metrics: raise ValueError('metrics must be a non-empty dictionary')
    record={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'name':name,'params':params or {},'metrics':metrics,'tags':tags or {}}
    dest=Path(path); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('a',encoding='utf-8') as f: f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    return record


def write_model_card(output_path: str, *, model_name: str, intended_use: str, data_description: str,
                     evaluation: str, limitations: str, owner: str = 'Harpreet Singh') -> str:
    fields={'Model name':model_name,'Owner':owner,'Intended use':intended_use,'Data description':data_description,
            'Evaluation':evaluation,'Limitations and risks':limitations}
    if any(not str(v).strip() for v in fields.values()): raise ValueError('All model-card fields must be filled in')
    body='# Model Card\n\n'+'\n\n'.join(f'## {k}\n{v}' for k,v in fields.items())+'\n'
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(body,encoding='utf-8')
    return str(path)


def read_experiments(path: str = 'experiments/runs.jsonl') -> list[dict]:
    """Read logged runs, skipping blank or corrupt lines."""
    src = Path(path)
    if not src.is_file():
        return []
    runs = []
    for line in src.read_text(encoding='utf-8').splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            runs.append(item)
    return runs


def best_experiment(metric: str, mode: str = 'max', path: str = 'experiments/runs.jsonl') -> dict | None:
    """Return the run with the best finite value of `metric` ('max' or 'min'); None if no run has it."""
    if mode not in ('max', 'min'):
        raise ValueError("mode must be 'max' or 'min'")
    candidates = []
    for run in read_experiments(path):
        value = run.get('metrics', {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float('inf'):
            candidates.append((value, run))
    if not candidates:
        return None
    pick = max if mode == 'max' else min
    return pick(candidates, key=lambda pair: pair[0])[1]

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--name',required=True); p.add_argument('--metrics',required=True,help='JSON object, e.g. \'{"accuracy":0.91}\''); p.add_argument('--params',default='{}'); p.add_argument('--path',default='experiments/runs.jsonl'); a=p.parse_args()
    print(json.dumps(log_experiment(a.name,json.loads(a.metrics),json.loads(a.params),path=a.path),indent=2))
