#!/usr/bin/env python3
"""Finish the pinned QA parquet using validated HTTP ranges, then verify SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import urllib.request

from download_materials import ROOT, sha256

p = argparse.ArgumentParser()
p.add_argument('--endpoint',default='https://hf-mirror.com')
p.add_argument('--workers',type=int,default=6)
a = p.parse_args()
spec = next(x for x in json.loads((ROOT/'configs/assets.lock.json').read_text())['assets'] if x['file']=='train.parquet')
output = ROOT/spec['destination']
if output.exists() and sha256(output)==spec['sha256']:
    print('Training data already verified')
    raise SystemExit(0)
part = output.with_name(output.name+'.part')
offset = part.stat().st_size if part.exists() else 0
size = spec['bytes']
chunks = output.parent/'train.ranges'
chunks.mkdir(exist_ok=True)
url = f"{a.endpoint}/datasets/{spec['repo']}/resolve/{spec['revision']}/train.parquet?download=true"

def fetch(bounds):
    start,end = bounds
    dest = chunks/f'{start}-{end}'
    if dest.exists() and dest.stat().st_size==end-start+1:
        return dest
    for attempt in range(6):
        try:
            request = urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as r:
                if r.status !=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                    raise IOError(f'Incorrect range response {r.status}: {r.headers.get("Content-Range")}')
                with dest.with_suffix('.part').open('wb') as f:
                    shutil.copyfileobj(r,f,1024*1024)
            if dest.with_suffix('.part').stat().st_size!=end-start+1:
                raise IOError('Incomplete range')
            dest.with_suffix('.part').replace(dest)
            print(f'Fetched range {start}-{end}',flush=True)
            return dest
        except OSError as e:
            if attempt==5:raise
            print(f'Range {start} retry: {e}',flush=True)
            time.sleep(min(30, 10+attempt*5) if getattr(e, 'code', None)==429 else 1+attempt)

bounds=[(start,min(start+4*1024*1024,size)-1) for start in range(offset,size,4*1024*1024)]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    list(pool.map(fetch,bounds))
assembled=output.with_name(output.name+'.assembling')
with assembled.open('wb') as f:
    if part.exists():
        with part.open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
    for start,end in bounds:
        with (chunks/f'{start}-{end}').open('rb') as source:shutil.copyfileobj(source,f,8*1024*1024)
if sha256(assembled)!=spec['sha256']:
    raise ValueError('Final checksum mismatch; partial and range files retained')
assembled.replace(output)
output.with_name(output.name+'.receipt.json').write_text(json.dumps({'url':url,'sha256':spec['sha256'],'bytes':size},indent=2)+'\n')
part.unlink(missing_ok=True)
shutil.rmtree(chunks)
print(f'Verified training data: {size} bytes',flush=True)
