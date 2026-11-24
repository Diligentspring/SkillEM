import copy
import json
import random

import pytest

from skillem.data import task_from_row, audit_splits, validate_training_tasks
from skillem.schema import FIELDS, Skill, Step, write_tasks, read_tasks
from skillem.search import SearchClient, evaluate, execution_prompt, official_score, rollout
from skillem.vendor.search_projection import search_projection
from skillem.vendor.search_metrics import compute_score


def test_official_scoring_and_projection_parity(task):
    actions = ['<think>...</think><answer>Paris</answer>',
               '<answer>Paris</answer><answer>London</answer>',
               '<search>0</search><answer>Paris</answer>', 'Paris', '<answer>the PARIS!</answer>']
    for action in actions:
        projected, _ = search_projection([action])
        assert official_score([Step(action)], ['Paris']) == compute_score(''.join(projected), {'target':['Paris']})
    assert official_score(task.steps, task.answers) == 1


def test_conditioning_and_projected_history(task):
    skill = Skill({f:'0' for f in FIELDS})
    assert skill.text() in execution_prompt(task, skill, [])
    assert skill.text() not in execution_prompt(task, skill, [], inject_initial=False)
    history = [Step('<think>private reasoning</think><search>0</search>', '<information>1</information>')]
    prompt = execution_prompt(task, skill, history)
    assert '<search>0</search>' in prompt
    assert '<information>1</information>' in prompt
    assert 'private reasoning' not in prompt


def test_search_wire_protocol_and_rollout(task, monkeypatch):
    import io
    received = []
    payload = {'result': [[{'document':{'contents':'  Evidence: Paris  '}, 'score':1.}]]}
    def fake_urlopen(request, timeout):
        received.append(json.loads(request.data))
        assert request.method == 'POST'
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    class ScriptedModel:
        def generate_action(self, task, skill, history, rng, temperature):
            assert task.answers == ['<withheld>']
            return ('<search>0</search>' if not history else '<answer>Paris</answer>'), None
    result = rollout(ScriptedModel(), task, None, SearchClient(), random.Random(0))
    assert result.verified
    assert result.steps[0].observation == '<information>{"result": "Doc 1: Evidence: Paris\\n"}</information>'
    assert received == [{'query':'0','topk':3,'return_scores':True}]
    assert evaluate([result])['nq']['em'] == 1.


def test_data_conversion_roundtrip_and_training_filter(tmp_path, task):
    row = {'question':'0','data_source':'nq','reward_model':{'ground_truth':{'target':['Paris']}}}
    converted = task_from_row(row,'train',0,'pinned-source')
    assert converted.answers == ['Paris']
    write_tasks(tmp_path/'data.jsonl',[task])
    assert read_tasks(tmp_path/'data.jsonl')[0].to_dict() == task.to_dict()
    validate_training_tasks([task])
    with pytest.raises(ValueError):
        validate_training_tasks([converted])
    heldout = copy.deepcopy(task)
    heldout.split = 'test'
    with pytest.raises(ValueError,match='leakage'):
        audit_splits({'train':[task],'test':[heldout]})


def test_collect_cli_saves_successes_incrementally(tmp_path, task, monkeypatch):
    from skillem.cli import main
    from skillem.schema import write_json
    class Model:
        calls = {}
        def generate_action(self, public, skill, history, rng, temperature):
            assert public.answers == ['<withheld>'] and not public.steps
            return '<answer>Paris</answer>', None
    monkeypatch.setattr('skillem.cli.load_backend',lambda *args,**kwargs:Model())
    write_tasks(tmp_path/'input.jsonl',[task])
    write_json(tmp_path/'config.json',{'model':{},'seed':0})
    args = ['collect','--input',str(tmp_path/'input.jsonl'),'--config',str(tmp_path/'config.json'),
            '--output',str(tmp_path/'out'),'--skill-mode','none']
    main(args)
    assert read_tasks(tmp_path/'out/successes.jsonl')[0].verified
    assert json.loads((tmp_path/'out/metrics.json').read_text())['nq']['em']==1.
    with pytest.raises(SystemExit):
        main(args)


def test_jsonl_preserves_unicode_line_separators(task, tmp_path):
    task.question = 'Question\u2028with\u0085Unicode separators'
    task.steps[0].observation = 'Evidence\u2029continues'
    path = tmp_path/'unicode.jsonl'
    write_tasks(path,[task])
    assert read_tasks(path)[0].to_dict()==task.to_dict()


def test_training_preparation_excludes_heldout_questions(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from skillem.data import prepare_parquet
    rows = [{'id':str(i),'question':q,'data_source':'nq','golden_answers':['Paris']}
            for i,q in enumerate(['keep 0','keep 1','held-out question','keep 2'])]
    pq.write_table(pa.Table.from_pylist(rows),tmp_path/'source.parquet')
    counts = prepare_parquet(tmp_path/'source.parquet',tmp_path/'prepared','pinned','train',
                             exclude_questions={'HELD-OUT QUESTION'})
    assert sum(counts.values())==3
    manifest = json.loads((tmp_path/'prepared/train.manifest.json').read_text())
    assert manifest['source_rows']==4
    assert manifest['excluded_overlap_ids']==['pinned:train:nq:2']
    with pytest.raises(ValueError,match='preserve the official test'):
        prepare_parquet(tmp_path/'source.parquet',tmp_path/'test','pinned','test',exclude_questions={'keep 0'})
