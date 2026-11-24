#!/usr/bin/env python3
"""Build a local random transformer and a synthetic training fixture; no network."""
from pathlib import Path
import json
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from skillem.model import HFModel
from skillem.schema import FIELDS, Task, Step, write_tasks, write_json

root = Path(os.environ.get('SKILLEM_SMOKE_ROOT', 'runs/smoke'))
torch.manual_seed(0)
vocab = {'<unk>':0,'<eos>':1,'<pad>':2,'User':3,':':4,'Assistant':5,
         '0':6,'1':7,'answer':8,'Paris':9,'<':10,'>':11,'/':12,'search':13}
t = Tokenizer(WordLevel(vocab,unk_token='<unk>'))
t.pre_tokenizer = Whitespace()
tok = PreTrainedTokenizerFast(tokenizer_object=t,unk_token='<unk>',eos_token='<eos>',pad_token='<pad>')
m = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=2048,n_ctx=2048,
            n_embd=16,n_layer=1,n_head=2,eos_token_id=1,pad_token_id=2))
model = HFModel(m,tok,max_field_tokens=2,max_action_tokens=8,max_context=2048)
model.save(root/'initial')
write_tasks(root/'successes.jsonl', [Task('synthetic-0','0',['Paris'],'nq','train',
            steps=[Step('<search>0</search>','<information>1</information>'),
                   Step('<answer>Paris</answer>')],verified=True,
            metadata={'synthetic_fixture':True, 'static_skill':{f:'0' for f in FIELDS}})])
base = {'seed':0,'rounds':2,'learning_rate':1e-3,'m_epochs':1,'batch_tasks':1,
        'model':{'name':str(root/'initial'),'device':'cpu','dtype':'float32',
                 'max_field_tokens':2,'max_action_tokens':8,'max_context':2048}}
for label in ('mcmc','variational','snis','prior_uniform','sft','static_skill',
              'is_em','is_em_mixture','is_em_learned'):
    method = 'is_em' if label.startswith('is_em') else label
    config = {**base,'method':method,'samples':2}
    if method=='mcmc':
        config['mcmc']={'chains':1,'burn_in':1,'samples_per_chain':2,'lazy_probability':.25,
                        'kernels':{'independent':.5,'local':.25,'mwg':.25}}
    if method=='variational':
        config['inference_model']={'name':str(root/'initial')}
        config['inference_learning_rate']=1e-3
    if method=='is_em':
        config.pop('samples')
        config['is_em']={'samples':2,'proposal':'prior','deduplicate':True}
        if label!='is_em':
            config['is_em'].update(proposal='mixture',prior_ratio=.25,
                                  proposal_epochs=1 if label=='is_em_learned' else 0)
            config['inference_model']={'name':str(root/'initial')}
            config['inference_learning_rate']=1e-3
    write_json(root/f'{label}.json',config)
print(f'Synthetic smoke assets prepared under {root}; these are not QA measurements.')
