#!/usr/bin/env python3
"""Write SkillEM's exact subsets into SkillRL's processed Parquet schema."""
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from skillem.schema import read_tasks
p = argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
a = p.parse_args()
rows = []
for task in read_tasks(a.input):
    ground = {'target':task.answers}
    rows.append({'data_source':task.data_source,
        'prompt':[{'role':'system','content':'You are a helpful and harmless assistant.'},
                  {'role':'user','content':task.question}],
        'ability':'fact-reasoning', 'reward_model':{'ground_truth':ground,'style':'rule'},
        'extra_info':{'index':task.task_id,'question':task.question,'split':task.split,'need_tools_kwargs':True,
            'tools_kwargs':{'search':{'create_kwargs':{'ground_truth':ground,'question':task.question,'data_source':task.data_source}}}},
        'env_kwargs':{'ground_truth':ground,'question':task.question,'data_source':task.data_source}})
if not rows:
    raise ValueError('Input is empty')
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows),a.output)
print(f'Wrote {len(rows)} tasks')
