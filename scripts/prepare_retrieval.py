#!/usr/bin/env python3
"""Assemble pinned E5 shards and decompress the Wikipedia corpus atomically."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--directory', default=str(root/'materials/retrieval'))
a = p.parse_args()
d = Path(a.directory)
lock = json.loads((root/'configs/assets.lock.json').read_text())
for filename in ('part_aa','part_ab','wiki-18.jsonl.gz'):
    spec = next(v for v in lock['assets'] if v['file']==filename)
    path = d/filename
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    if h.hexdigest() != spec['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
for output in ('e5_Flat.index','wiki-18.jsonl'):
    if (d/output).exists():
        print(f'Already exists: {output}')
        continue
    temp = d/(output+'.part')
    with temp.open('wb') as dest:
        if output == 'e5_Flat.index':
            for name in ('part_aa','part_ab'):
                with (d/name).open('rb') as source:
                    shutil.copyfileobj(source,dest,8*1024*1024)
        else:
            with gzip.open(d/'wiki-18.jsonl.gz','rb') as source:
                shutil.copyfileobj(source,dest,8*1024*1024)
    temp.replace(d/output)
    print(f'Prepared: {output}')
