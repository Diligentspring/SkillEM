#!/usr/bin/env python3
"""Report local pinned materials without treating partial downloads as assets."""
import json
from pathlib import Path
from download_materials import ROOT, sha256

lock = json.loads((ROOT/'configs/assets.lock.json').read_text())
records = []
for spec in lock['assets']:
    path = ROOT/spec['destination']
    record = {k:spec[k] for k in ('group','repo','revision','file','destination')}
    record['expected_bytes'] = spec.get('bytes')
    if path.exists():
        digest = sha256(path)
        verified = digest==spec['sha256'] if spec.get('sha256') else path.stat().st_size==spec.get('bytes')
        record.update(status='verified' if verified else 'invalid',bytes=path.stat().st_size,sha256=digest,
                      verification='published_lfs_sha256' if spec.get('sha256') else 'pinned_git_file_size')
    else:
        part = path.with_name(path.name+'.part')
        record.update(status='partial' if part.exists() else 'not_downloaded',
                      partial_bytes=part.stat().st_size if part.exists() else 0)
    records.append(record)
source = json.loads((ROOT/'materials/skillrl_source.json').read_text())
result = {'source':source,'assets':records,
          'large_assets_note':'7B model tensors and full Wikipedia index are server downloads; local QA training has not been run.'}
(ROOT/'materials/status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({state:sum(r['status']==state for r in records) for state in ('verified','partial','invalid','not_downloaded')}))
