"""Exact finite-space and real-transformer checks for IS-EM."""
import copy
import json
import math
import random
from collections import Counter

import pytest
import torch

from skillem.finite import FiniteModel, finite_tasks
from skillem.inference import (PosteriorSample, exact_posterior, importance_posterior,
                               sample_is)
from skillem.math_utils import categorical, softmax
from skillem.model import HFModel
from skillem.proposals import (ISConfig, MixtureProposal, PriorProposal, is_config,
                               needs_inference_model)
from skillem.schema import FIELDS, Skill, write_json, write_tasks
from skillem.training import make_optimizer, m_step, train


class FiniteInference:
    def __init__(self, states, probabilities=(.7, .1, .1, .1)):
        self.states, self.probabilities = states, probabilities

    def sample_inference(self, task, rng):
        assert 'outcome' in task.metadata
        return self.states[categorical(rng, self.probabilities)]

    def log_inference(self, task, z):
        p = self.probabilities[next(i for i,s in enumerate(self.states) if s.key()==z.key())]
        return math.log(p) if p else -math.inf


def clone_backend(model):
    return HFModel(copy.deepcopy(model.model), model.tokenizer,
                   max_field_tokens=model.max_field_tokens,
                   max_action_tokens=model.max_action_tokens,
                   max_context=model.max_context)


def test_prior_reproduces_legacy_snis_exactly():
    model, task = FiniteModel(), finite_tasks()[0]
    expected_rng, rng = random.Random(42), random.Random(42)
    states = [model.sample_prior(task.public(),expected_rng) for _ in range(25)]
    weights = softmax([model.log_execution(task,z) for z in states])
    samples, diagnostics, records = sample_is(model,task,ISConfig(samples=25),rng,
                                              theta_version='theta-0')
    assert [s.skill.key() for s in samples]==[s.key() for s in states]
    assert [s.weight for s in samples]==weights
    assert rng.getstate()==expected_rng.getstate()
    assert diagnostics['ess_ratio']==pytest.approx(diagnostics['ess']/25)
    assert all(r['log_weight']==r['log_execution'] for r in records)
    assert all(r['theta_version']=='theta-0' for r in records)
    legacy, _ = importance_posterior(model,task,25,random.Random(42))
    assert [s.to_dict() for s in legacy]==[s.to_dict() for s in samples]


def test_mixture_uses_full_density_for_both_components():
    model, task = FiniteModel(), finite_tasks()[0]
    phi = FiniteInference(model.states)
    proposal = MixtureProposal(model,phi,.3)
    assert sum(math.exp(proposal.log_prob(task,z)) for z in model.states)==pytest.approx(1.)
    _, _, records = sample_is(model,task,ISConfig(samples=80,proposal='mixture',prior_ratio=.3),
                              random.Random(8),phi,theta_version='theta-2',inference_version='phi-2')
    assert {r['proposal_component'] for r in records}=={'prior','inference'}
    for r in records:
        expected = math.log(.3*math.exp(r['log_prior'])+.7*math.exp(r['log_inference']))
        assert r['log_proposal']==pytest.approx(expected)
        assert r['log_weight']==pytest.approx(r['log_prior']+r['log_execution']-expected)
        assert r['inference_version']=='phi-2'
    assert len({r['proposal_version'] for r in records})==1


@pytest.mark.parametrize('kind',['prior','mixture'])
def test_is_recovers_exact_posterior_and_evidence(kind):
    model, task = FiniteModel(), finite_tasks()[0]
    phi = FiniteInference(model.states)
    options = ISConfig(samples=20000,proposal=kind,prior_ratio=.3,deduplicate=True)
    samples, diag, _ = sample_is(model,task,options,random.Random(12),phi)
    exact, log_z = exact_posterior(model,task,model.states)
    estimated = {s.skill.key():s.weight for s in samples}
    tv = .5*sum(abs(s.weight-estimated.get(s.skill.key(),0.)) for s in exact)
    assert tv<.025
    assert diag['log_Z_hat']==pytest.approx(log_z,abs=.025)
    assert diag['unique_skills']==4


def test_duplicate_aggregation_preserves_mass_and_diagnostics():
    model, task = FiniteModel(), finite_tasks()[0]
    raw, diag, records = sample_is(model,task,ISConfig(samples=40),random.Random(9))
    unique, other_diag, other_records = sample_is(model,task,ISConfig(samples=40,deduplicate=True),random.Random(9))
    mass = Counter()
    for sample in raw:
        mass[sample.skill.key()]+=sample.weight
    assert {s.skill.key():s.weight for s in unique}==dict(mass)
    assert diag==other_diag and records==other_records
    assert diag['duplicate_rate']>.5


def test_prior_channel_covers_zero_mass_inference_component():
    model, task = FiniteModel(), finite_tasks()[0]
    phi = FiniteInference(model.states,(1.,0.,0.,0.))
    samples, diag, records = sample_is(model,task,
        ISConfig(samples=100,proposal='mixture',prior_ratio=.4),random.Random(1),phi)
    assert all(math.isfinite(r['log_proposal']) for r in records)
    assert any(r['log_inference'] is None for r in records)
    assert sum(s.weight for s in samples)==pytest.approx(1.)
    json.dumps(records,allow_nan=False)


def test_invalid_scores_and_zero_weights_fail():
    model, task = FiniteModel(), finite_tasks()[0]
    model.log_execution = lambda *a:-math.inf
    with pytest.raises(ValueError,match='zero mass'):
        sample_is(model,task,ISConfig(),random.Random(0))
    model.log_execution = lambda *a:math.nan
    with pytest.raises(ValueError,match='log probability'):
        sample_is(model,task,ISConfig(),random.Random(0))
    model.log_execution = lambda *a:-2.
    model.log_prior = lambda *a:-math.inf
    with pytest.raises(ValueError,match='zero proposal probability'):
        sample_is(model,task,ISConfig(),random.Random(0))


@pytest.mark.parametrize('options',[{'samples':0},{'samples':True},{'samples':2.5},
    {'proposal':'black_box'},{'proposal':'mixture','prior_ratio':0.},
    {'proposal':'mixture','prior_ratio':1.},{'prior_ratio':float('nan')},
    {'deduplicate':1},{'proposal_epochs':-1},{'proposal_epochs':1}])
def test_is_config_validation(options):
    with pytest.raises(ValueError):
        ISConfig(**options).validate()


def test_config_alias_and_optional_phi_loading():
    assert is_config({'method':'snis'}).samples==4
    assert is_config({'method':'is_em','samples':8}).samples==8
    assert is_config({'method':'is_em','is_em':{'samples':12}}).samples==12
    assert not needs_inference_model({'method':'is_em'})
    assert needs_inference_model({'method':'is_em','is_em':{'proposal':'mixture'}})
    with pytest.raises(ValueError,match='prior-only'):
        is_config({'method':'snis','is_em':{'proposal':'mixture'}})
    with pytest.raises(ValueError,match='configuration'):
        is_config({'method':'is_em','is_em':{'unknown':1}})


def test_is_e_step_and_frozen_proposal_do_not_mutate_parameters(tiny_model,task,tmp_path):
    phi = clone_backend(tiny_model)
    old_phi = {k:v.clone() for k,v in phi.model.state_dict().items()}
    config = {'method':'is_em','rounds':1,'learning_rate':1e-3,
              'is_em':{'samples':3,'proposal':'mixture','prior_ratio':.4}}
    history = train(tiny_model,[task],config,tmp_path,phi)
    assert all(torch.equal(v,phi.model.state_dict()[k]) for k,v in old_phi.items())
    assert all(p.grad is None and not p.requires_grad for p in phi.model.parameters())
    assert (tmp_path/'round-001/phi/skillem_model.json').exists()
    records = json.loads((tmp_path/'round-001/trace.json').read_text())[task.task_id]
    assert len(records)==3
    assert sum(r['normalized_weight'] for r in records)==pytest.approx(1.)
    assert 'inference_model_calls' in history[0]
    assert history[0]['is_em']['samples_per_task']==3


def test_is_weights_produce_joint_gradient(tiny_model,task):
    phi = clone_backend(tiny_model)
    with torch.no_grad():
        samples, _, _ = sample_is(tiny_model,task,
            ISConfig(samples=3,proposal='mixture'),random.Random(3),phi)
    loss = sum(-s.weight*(tiny_model.log_prior(task,s.skill)+tiny_model.log_execution(task,s.skill))
               for s in samples)
    loss.backward()
    expected = [p.grad.clone() for p in tiny_model.model.parameters()]
    m_step(tiny_model,[task],{task.task_id:samples},torch.optim.SGD(tiny_model.model.parameters(),lr=0.),
           max_grad_norm=1e9)
    for p, grad in zip(tiny_model.model.parameters(),expected):
        assert torch.allclose(p.grad,grad,atol=2e-5,rtol=2e-4)
    assert all(p.grad is None for p in phi.model.parameters())


def test_proposal_fit_is_weighted_cross_entropy_only(tiny_model,task):
    z0 = tiny_model.encode_skill({f:'0' for f in FIELDS})
    z1 = tiny_model.encode_skill({f:'1' for f in FIELDS})
    samples = [PosteriorSample(z0,.7),PosteriorSample(z1,.3)]
    expected_loss = sum(-s.weight*tiny_model.log_inference(task,s.skill) for s in samples)
    expected_loss.backward()
    expected = [p.grad.clone() for p in tiny_model.model.parameters()]
    def forbidden(*a):
        raise AssertionError('Proposal fitting must not train execution')
    tiny_model.iter_execution_scores = forbidden
    metrics = m_step(tiny_model,[task],{task.task_id:samples},
        torch.optim.SGD(tiny_model.model.parameters(),lr=0.),max_grad_norm=1e9,objective='proposal')
    assert 'weighted_proposal_nll' in metrics[0]
    for p,grad in zip(tiny_model.model.parameters(),expected):
        assert torch.allclose(p.grad,grad,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('proposal,epochs',[('prior',0),('mixture',0),('mixture',1)])
def test_is_resume_matches_uninterrupted_parameters(tiny_model,task,tmp_path,monkeypatch,proposal,epochs):
    import skillem.training as module
    config = {'method':'is_em','rounds':2,'seed':23,'learning_rate':1e-3,
              'inference_learning_rate':1e-3,
              'is_em':{'samples':2,'proposal':proposal,'proposal_epochs':epochs,'deduplicate':True}}
    full_theta = clone_backend(tiny_model)
    full_phi = clone_backend(tiny_model) if proposal=='mixture' else None
    train(full_theta,[task],config,tmp_path/'full',full_phi)
    other_theta = clone_backend(tiny_model)
    other_phi = clone_backend(tiny_model) if proposal=='mixture' else None
    original, calls = module.m_step, 0
    def interrupt(model,*args,**kwargs):
        nonlocal calls
        if model is other_theta:
            calls+=1
            if calls==2:
                raise RuntimeError('simulated interruption')
        return original(model,*args,**kwargs)
    monkeypatch.setattr(module,'m_step',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        train(other_theta,[task],config,tmp_path/'interrupted',other_phi)
    monkeypatch.setattr(module,'m_step',original)
    checkpoint = tmp_path/'interrupted/round-001'
    loaded = HFModel.load(str(checkpoint/'theta'))
    loaded_phi = HFModel.load(str(checkpoint/'phi')) if other_phi else None
    train(loaded,[task],config,tmp_path/'interrupted',loaded_phi,resume=True)
    assert all(torch.equal(v,loaded.model.state_dict()[k]) for k,v in full_theta.model.state_dict().items())
    if full_phi:
        assert all(torch.equal(v,loaded_phi.model.state_dict()[k]) for k,v in full_phi.model.state_dict().items())
        if epochs:
            assert any(not torch.equal(v,tiny_model.model.state_dict()[k]) for k,v in full_phi.model.state_dict().items())
            metrics = json.loads((tmp_path/'full/round-001/metrics.json').read_text())
            assert metrics['proposal_fit_target']==metrics['is_em']['theta_version']
    full_trace = json.loads((tmp_path/'full/round-002/trace.json').read_text())
    resumed_trace = json.loads((tmp_path/'interrupted/round-002/trace.json').read_text())
    assert full_trace==resumed_trace


@pytest.mark.parametrize('proposal',['prior','mixture'])
def test_is_cli_loads_and_restores_only_required_models(tiny_model,task,tmp_path,monkeypatch,proposal):
    from skillem.cli import main
    initial = tmp_path/'initial'
    tiny_model.save(initial)
    config = {'method':'is_em','rounds':1,'model':{'name':str(initial)},
              'is_em':{'samples':2,'proposal':proposal}}
    write_tasks(tmp_path/'tasks.jsonl',[task])
    write_json(tmp_path/'config.json',config)
    args = ['train','--config',str(tmp_path/'config.json'),'--input',str(tmp_path/'tasks.jsonl'),
            '--output',str(tmp_path/'run')]
    main(args)
    latest = json.loads((tmp_path/'run/latest.json').read_text())
    assert latest['round']==1
    assert (tmp_path/'run/round-001/phi').exists()==(proposal=='mixture')
    main(args+['--resume'])


def test_bad_mixture_configuration_fails_before_model_loading(tmp_path,task,monkeypatch):
    from skillem.cli import main
    write_tasks(tmp_path/'tasks.jsonl',[task])
    write_json(tmp_path/'config.json',{'method':'is_em','model':{},
               'is_em':{'proposal':'mixture','prior_ratio':0.}})
    def forbidden(*a,**kw):
        raise AssertionError('Invalid options should be checked before model loading')
    monkeypatch.setattr('skillem.cli.load_backend',forbidden)
    with pytest.raises(ValueError,match='prior_ratio'):
        main(['train','--config',str(tmp_path/'config.json'),'--input',str(tmp_path/'tasks.jsonl'),
              '--output',str(tmp_path/'out')])


def test_training_rejects_shared_parameters_and_missing_phi(tiny_model,task,tmp_path):
    config = {'method':'is_em','is_em':{'proposal':'mixture'}}
    with pytest.raises(ValueError,match='independent inference model'):
        train(tiny_model,[task],config,tmp_path)
    with pytest.raises(ValueError,match='independent parameters'):
        train(tiny_model,[task],config,tmp_path,tiny_model)
