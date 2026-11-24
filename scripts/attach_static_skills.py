#!/usr/bin/env python3
import argparse
from skillem.bank import StaticSkillBank
from skillem.schema import read_tasks, write_tasks
p = argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
p.add_argument('--bank',required=True)
a = p.parse_args()
bank = StaticSkillBank(a.bank)
tasks = read_tasks(a.input)
for task in tasks:
    task.metadata['static_skill'] = bank.retrieve(task.question).fields
write_tasks(a.output,tasks)
