#!/usr/bin/env python3
"""Summarize completed local checks, without claiming GPU QA results."""
import argparse
import json
import re
from pathlib import Path
from skillem.schema import write_json

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--smoke-root',default='runs/smoke-verified')
p.add_argument('--date',help='Client-local validation date, YYYY-MM-DD')
a=p.parse_args()
log=(root/'materials/pytest-validation.log').read_text()
match=re.search(r'(\d+) passed',log)
if not match or 'failed' in log:
    raise ValueError('No successful pytest log')
methods={}
for method in ('mcmc','variational','snis','prior_uniform','sft','static_skill',
               'is_em','is_em_mixture','is_em_learned'):
    run=root/a.smoke_root/method
    latest=json.loads((run/'latest.json').read_text())
    if latest['round']!=2:
        raise ValueError(f'Incomplete smoke run: {method}')
    methods[method]={'completed_rounds':2,
        'metrics':[json.loads((run/f'round-{i:03d}/metrics.json').read_text()) for i in (1,2)],
        'scope':'synthetic fixture and randomly initialized tiny Transformer'}
finite=json.loads((root/'runs/finite-demo.json').read_text())
report={'date':a.date,'timezone':'Asia/Shanghai','pytest_passed':int(match.group(1)),
    'checks':['finite posterior recovery for all four kernels','local detailed balance',
        'exact finite EM likelihood','sequence-sum loss and prompt/observation masking',
        'joint M-step gradient parity','leave-one-out gradient expectation',
        'theta frozen during variational E-step','Qwen2 LoRA update/save/reload',
        'interrupted resume matches uninterrupted parameters','validation best checkpoint and early stopping',
        'official scorer/projection parity','incremental CLI collection',
        'JSONL Unicode separators','held-out question exclusion',
        'IS prior regression against legacy SNIS','full mixture denominator and exact posterior recovery',
        'IS joint gradient and proposal-only weighted cross entropy',
        'IS prior/frozen-mixture/learned-mixture interrupted resume parity'],
    'smoke_runs':methods,'finite_audit':finite,
    'not_run':['7B GPU training','full Wikipedia E5 retrieval','full seven-benchmark QA evaluation',
               'large-model convergence certification','distributed verl/vLLM training']}
audit=root/'materials/split_audit.json'
if audit.exists():
    report['data_audit']=json.loads(audit.read_text())
write_json(root/'materials/verification.json',report)
print(f"Recorded {report['pytest_passed']} tests and {len(methods)} two-round training paths")
