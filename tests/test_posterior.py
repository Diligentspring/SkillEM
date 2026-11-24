import math
import random

import pytest

from skillem.finite import FiniteModel, finite_tasks
from skillem.inference import (MCMCConfig, exact_posterior, sample_mcmc,
                               mh_log_accept, importance_posterior)
from skillem.math_utils import loo_advantages, softmax


def test_mh_correction_and_impossible_reverse():
    assert mh_log_accept(-5., -4., -1., -3.) == -1.
    assert mh_log_accept(-5., -4., -1., -math.inf) == -math.inf
    with pytest.raises(ValueError):
        mh_log_accept(-5., -4., -math.inf, -1.)


@pytest.mark.parametrize('kernel', ['independent', 'local', 'mwg', 'gibbs', 'mixed'])
def test_chains_recover_enumerated_posterior(kernel):
    model, task = FiniteModel(), finite_tasks()[0]
    weights = {'independent': 1.} if kernel == 'independent' else {'independent': .3, kernel: .7}
    if kernel == 'mixed':
        weights = {'independent': .3, 'local': .3, 'mwg': .2, 'gibbs': .2}
    samples, diag, trace = sample_mcmc(model, task,
        MCMCConfig(chains=2, burn_in=300, samples_per_chain=12000, kernels=weights),
        random.Random(42), finite_domains=model.domains)
    exact, _ = exact_posterior(model, task, model.states)
    estimated = {s.skill.key(): s.weight for s in samples}
    tv = .5 * sum(abs(s.weight - estimated.get(s.skill.key(), 0)) for s in exact)
    assert tv < .035
    assert diag['retained_states'] == 24000
    assert sum(s.weight for s in samples) == pytest.approx(1.)
    assert any(not t['accepted'] for t in trace)


def test_rejections_and_lazy_states_retain_frequency():
    model, task = FiniteModel(), finite_tasks()[0]
    samples, diag, trace = sample_mcmc(model, task,
        MCMCConfig(chains=1, burn_in=0, samples_per_chain=1000, lazy_probability=.95), random.Random(3))
    from collections import Counter
    counts = Counter(t['state'] for t in trace)
    for sample in samples:
        assert sample.weight == counts[sample.skill.key()] / 1000
    assert diag['retained_states'] == 1000


def test_exact_em_marginal_likelihood_increases():
    model, tasks = FiniteModel(), finite_tasks()
    def objective():
        return sum(exact_posterior(model, t, model.states)[1] for t in tasks)
    before = objective()
    for _ in range(4):
        model.exact_em_update(tasks)
        after = objective()
        assert after >= before - 1e-12
        before = after


def test_importance_and_leave_one_out():
    assert loo_advantages([1., 3., 5.]) == [-3., 0., 3.]
    with pytest.raises(ValueError):
        loo_advantages([1.])
    assert softmax([-10000., -10000.]) == pytest.approx([.5, .5])
    model = FiniteModel()
    samples, diag = importance_posterior(model, finite_tasks()[0], 100, random.Random(0))
    assert sum(s.weight for s in samples) == pytest.approx(1.)
    assert 1 <= diag['ess'] <= 100


def test_local_detailed_balance_on_all_edges():
    model, task = FiniteModel(), finite_tasks()[0]
    posterior, _ = exact_posterior(model, task, model.states)
    pi = {s.skill.key(): s.weight for s in posterior}
    from skillem.inference import joint
    for z in model.states:
        for field in model.domains:
            for value in model.domains[field]:
                other = z.replace(field, value)
                f = model.log_field(task, z, other, field, 'local')
                r = model.log_field(task, other, z, field, 'local')
                a = mh_log_accept(joint(model, task, z), joint(model, task, other), f, r)
                b = mh_log_accept(joint(model, task, other), joint(model, task, z), r, f)
                assert pi[z.key()] * math.exp(f+a) == pytest.approx(pi[other.key()] * math.exp(r+b))
