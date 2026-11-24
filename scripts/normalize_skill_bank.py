#!/usr/bin/env python3
"""Create a separately named schema-corrected SkillRL Search bank."""
import argparse
import json
from pathlib import Path
from skillem.schema import write_json

p = argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
a = p.parse_args()
if Path(a.input).resolve()==Path(a.output).resolve():
    raise ValueError('Preserve the original bank; use a distinct output path')
bank = json.loads(Path(a.input).read_text())
if 'query_type_skills' in bank:
    if 'task_specific_skills' in bank and bank['task_specific_skills']!=bank['query_type_skills']:
        raise ValueError('Conflicting category schemas; resolve explicitly')
    bank['task_specific_skills'] = bank['query_type_skills']
write_json(a.output,bank)
print('Wrote a schema-corrected copy; report it separately from the released bank baseline.')
