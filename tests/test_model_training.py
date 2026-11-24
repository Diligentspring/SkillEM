import itertools
import json
import math
import random

import pytest
import torch

from skillem.model import HFModel, ContextOverflow
from skillem.schema import FIELDS, Skill, Task
from skillem.inference import PosteriorSample
from skillem.training import m_step, make_optimizer, variational_e_step, train


def test_bounded_distribution_normalizes_and_preserves_token_state(tiny_model, task):
    tiny_model.max_field_tokens = 1
    prefix = tiny_model._field_prompt(task.public(), FIELDS[0])
    allowed = [i for i in range(tiny_model.model.config.vocab_size) if i not in tiny_model.banned]
    states = [(tiny_model.eos,)] + [(i, tiny_model.eos) for i in allowed if i != tiny_model.eos]
    with torch.no_grad():
        total = sum(math.exp(float(tiny_model.sequence_logprob(prefix, z, bounded=True))) for z in states)
    assert total == pytest.approx(1., abs=1e-6)
    skill = tiny_model.sample_prior(task, random.Random(0))
    assert Skill.from_dict(json.loads(json.dumps(skill.to_dict()))).key() == skill.key()
    assert all(ids[-1] == tiny_model.eos for ids in skill.token_ids.values())


def test_sequence_sum_and_prompt_mask(tiny_model):
    prefix, target = [3, 4, 5], [6, 7, 1]
    actual = tiny_model.sequence_logprob(prefix, target)
    ids = torch.tensor([prefix + target])
    logits = tiny_model.model(ids).logits
    expected = sum(torch.log_softmax(logits[0, len(prefix)-1+i], -1)[token] for i, token in enumerate(target))
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, expected / len(target))
    assert torch.allclose(tiny_model.sequence_logprob(prefix, [6,1], bounded=True),
                          tiny_model.sequence_logprob(prefix, [6,1], bounded=True))
    with pytest.raises(ContextOverflow):
        tiny_model.sequence_logprob([3] * tiny_model.max_context, [1])


def test_prior_cannot_see_gold_or_success_trajectory(tiny_model, task):
    import copy
    z = tiny_model.encode_skill({f:'0' for f in FIELDS})
    changed = copy.deepcopy(task)
    changed.answers, changed.steps = ['not visible'], []
    assert torch.allclose(tiny_model.log_prior(task, z), tiny_model.log_prior(changed, z))


def test_m_step_weights_match_joint_gradient(tiny_model, task):
    a = tiny_model.encode_skill({f:'0' for f in FIELDS})
    b = tiny_model.encode_skill({f:'1' for f in FIELDS})
    samples = [PosteriorSample(a, .8), PosteriorSample(b, .2)]
    expected = sum(-s.weight * (tiny_model.log_prior(task, s.skill) + tiny_model.log_execution(task, s.skill))
                   for s in samples)
    expected.backward()
    reference = [p.grad.detach().clone() for p in tiny_model.model.parameters()]
    optimizer = torch.optim.SGD(tiny_model.model.parameters(), lr=0.)
    m_step(tiny_model, [task], {task.task_id:samples}, optimizer, max_grad_norm=1e9)
    for p, expected_grad in zip(tiny_model.model.parameters(), reference):
        assert torch.allclose(p.grad, expected_grad, atol=2e-5, rtol=2e-4)


def test_variational_freezes_theta_and_updates_phi(tiny_model, task):
    import copy
    phi = HFModel(copy.deepcopy(tiny_model.model), tiny_model.tokenizer,
                  max_field_tokens=2, max_action_tokens=8, max_context=2048)
    before_theta = [p.detach().clone() for p in tiny_model.model.parameters()]
    before_phi = [p.detach().clone() for p in phi.model.parameters()]
    optimizer = make_optimizer(phi, 1e-3)
    metrics = variational_e_step(tiny_model, phi, [task], optimizer, random.Random(8), count=3)
    assert all(torch.equal(a, p) for a, p in zip(before_theta, tiny_model.model.parameters()))
    assert all(p.grad is None for p in tiny_model.model.parameters())
    assert any(not torch.equal(a,p) for a,p in zip(before_phi,phi.model.parameters()))
    assert math.isfinite(metrics[0]['elbo_sample_mean_before_updates'])


def test_training_round_save_and_resume(tiny_model, task, tmp_path):
    config = {'method':'mcmc', 'seed':0, 'rounds':2, 'learning_rate':1e-4,
              'mcmc':{'chains':1,'burn_in':1,'samples_per_chain':2,'lazy_probability':0.}}
    history = train(tiny_model, [task], config, tmp_path)
    assert len(history) == 2
    latest = json.loads((tmp_path/'latest.json').read_text())
    loaded = HFModel.load(str(tmp_path/latest['round_dir']/'theta'), device='cpu')
    loaded.resume_loaded = True
    assert loaded.max_field_tokens == tiny_model.max_field_tokens
    z = tiny_model.encode_skill({f:'0' for f in FIELDS})
    assert torch.allclose(tiny_model.log_prior(task,z),loaded.log_prior(task,z))
    resumed = train(loaded, [task], config, tmp_path, resume=True)
    assert resumed == history
    with pytest.raises(ValueError,match='same resolved config'):
        train(loaded,[task],{**config,'seed':1},tmp_path,resume=True)


def test_resumed_update_matches_uninterrupted_run(tiny_model, task, tmp_path, monkeypatch):
    import copy
    import skillem.training as module
    initial = copy.deepcopy(tiny_model.model.state_dict())
    config = {'method':'mcmc','seed':9,'rounds':2,'learning_rate':1e-3,
              'mcmc':{'chains':1,'burn_in':1,'samples_per_chain':2,'lazy_probability':0.}}
    train(tiny_model,[task],config,tmp_path/'complete')
    expected = {k:v.clone() for k,v in tiny_model.model.state_dict().items()}
    tiny_model.model.load_state_dict(initial)
    original_mstep = module.m_step
    calls = 0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return original_mstep(*args,**kwargs)
    monkeypatch.setattr(module,'m_step',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        train(tiny_model,[task],config,tmp_path/'interrupted')
    monkeypatch.setattr(module,'m_step',original_mstep)
    loaded = HFModel.load(str(tmp_path/'interrupted/round-001/theta'))
    loaded.resume_loaded = True
    train(loaded,[task],config,tmp_path/'interrupted',resume=True)
    assert all(torch.equal(expected[k], v) for k,v in loaded.model.state_dict().items())


def test_qwen_lora_save_reload_and_training(tiny_model, task, tmp_path):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=len(tiny_model.tokenizer),hidden_size=16,intermediate_size=32,
                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                      max_position_embeddings=2048,eos_token_id=1,pad_token_id=2)
    base = tmp_path/'qwen-base'
    Qwen2ForCausalLM(cfg).save_pretrained(base)
    tiny_model.tokenizer.save_pretrained(base)
    model = HFModel.load(str(base),lora_rank=2,gradient_checkpointing=True,max_field_tokens=2,
                         max_context=2048,max_action_tokens=8)
    z = model.encode_skill({f:'0' for f in FIELDS})
    opt = make_optimizer(model,1e-3)
    m_step(model,[task],{task.task_id:[PosteriorSample(z,1.)]},opt)
    model.save(tmp_path/'adapter')
    reloaded = HFModel.load(str(tmp_path/'adapter'),gradient_checkpointing=True)
    assert torch.allclose(model.log_prior(task,z),reloaded.log_prior(task,z),atol=1e-5)
    assert any(p.requires_grad for p in reloaded.model.parameters())


def test_validation_best_checkpoint_and_early_stop(tiny_model, task, tmp_path):
    config = {'method':'sft','rounds':5,'seed':0,'patience':1,'learning_rate':1e-3}
    scores = iter([.5,.4])
    def validation(model, iteration):
        assert not torch.is_grad_enabled()
        return {'macro_em':next(scores)}
    history = train(tiny_model,[task],config,tmp_path,validation_fn=validation)
    assert len(history)==2
    assert json.loads((tmp_path/'best.json').read_text())['round']==1
    assert json.loads((tmp_path/'latest.json').read_text())['early_stopped']


def test_loo_score_function_is_unbiased_on_enumerated_pairs():
    from skillem.math_utils import loo_advantages
    logits = torch.tensor([.2,-.3],requires_grad=True)
    log_target = torch.tensor([-.7,-2.])
    p = torch.softmax(logits,0)
    logq = torch.log_softmax(logits,0)
    elbo = (p*(log_target-logq)).sum()
    expected, = torch.autograd.grad(elbo,logits,retain_graph=True)
    surrogate = 0.
    for i,j in itertools.product(range(2),repeat=2):
        rewards = [(log_target-logq)[i].detach().item(),(log_target-logq)[j].detach().item()]
        advantage = loo_advantages(rewards)
        surrogate = surrogate + (p[i]*p[j]).detach()*(advantage[0]*logq[i]+advantage[1]*logq[j])/2
    actual, = torch.autograd.grad(surrogate,logits)
    assert torch.allclose(actual,expected,atol=1e-7)
