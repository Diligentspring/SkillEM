"""Posterior kernels. Scores are full sequence sums, never length averages."""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .math_utils import categorical, logsumexp, softmax, weight_diagnostics, chain_diagnostics
from .schema import FIELDS, Skill, Task


def joint(model, task, skill):
    return float(model.log_prior(task, skill)) + float(model.log_execution(task, skill))


def mh_log_accept(log_current, log_candidate, log_forward, log_reverse):
    if any(math.isnan(x) or x == math.inf for x in
           (log_current, log_candidate, log_forward, log_reverse)):
        raise ValueError("Invalid MH score")
    if log_current == -math.inf or log_forward == -math.inf:
        raise ValueError("Current target and sampled proposal must have positive mass")
    if log_reverse == -math.inf or log_candidate == -math.inf:
        return -math.inf
    return min(0., log_candidate - log_current + log_reverse - log_forward)


@dataclass
class PosteriorSample:
    skill: Skill
    weight: float

    def to_dict(self):
        return {"skill": self.skill.to_dict(), "weight": self.weight}


@dataclass
class MCMCConfig:
    chains: int = 2
    burn_in: int = 8
    samples_per_chain: int = 4
    thin: int = 1
    lazy_probability: float = .5
    kernels: dict[str, float] = field(default_factory=lambda: {"independent": 1.})

    def validate(self):
        if self.chains < 1 or self.burn_in < 0 or self.samples_per_chain < 1 or self.thin < 1:
            raise ValueError("Invalid chain budgets")
        if not 0 <= self.lazy_probability < 1:
            raise ValueError("Invalid laziness")
        if (not self.kernels or any(k not in {"independent", "local", "mwg", "gibbs"} for k in self.kernels)
                or any(v < 0 for v in self.kernels.values())
                or not math.isclose(sum(self.kernels.values()), 1.)):
            raise ValueError("Kernel probabilities must be nonnegative and sum to one")
        if self.kernels.get("independent", 0) <= 0:
            raise ValueError("Retain a full-support independent proposal channel")


def exact_posterior(model, task: Task, states: list[Skill]):
    if not states or len({s.key() for s in states}) != len(states):
        raise ValueError("Exact support must be nonempty and unique")
    # Caller defines the complete finite support; this is not a sampled shortlist.
    scores = [joint(model, task, z) for z in states]
    return [PosteriorSample(z, w) for z, w in zip(states, softmax(scores))], logsumexp(scores)


def sample_mcmc(model, task: Task, config: MCMCConfig, rng: random.Random,
                initial: list[Skill] | None = None, finite_domains: dict | None = None):
    config.validate()
    if initial is not None and len(initial) != config.chains:
        raise ValueError("Need one initial state per chain")
    if config.kernels.get("gibbs", 0) and not finite_domains:
        raise ValueError("Exact Gibbs needs the model's full finite field domains")
    counters, chains, trace = Counter(), [], []
    # Cache is scoped to one frozen E-step and discarded after a model update.
    cache = {}

    def score(z):
        key = z.key()
        if key not in cache:
            cache[key] = joint(model, task, z)
            counters["joint_evaluations"] += 1
        return cache[key]

    names, probs = list(config.kernels), list(config.kernels.values())
    for chain_id in range(config.chains):
        state = initial[chain_id] if initial else model.sample_prior(task.public(), rng)
        current = score(state)
        if not math.isfinite(current):
            raise ValueError("Initialize a chain in the target support")
        kept = []
        transitions = config.burn_in + config.samples_per_chain * config.thin
        for step in range(transitions):
            kernel, accepted, log_alpha = "lazy", False, None
            if rng.random() >= config.lazy_probability:
                kernel = names[categorical(rng, probs)]
                counters[kernel + "/attempts"] += 1
                if kernel == "independent":
                    candidate = model.sample_prior(task.public(), rng)
                    forward = float(model.log_prior(task, candidate))
                    reverse = float(model.log_prior(task, state))
                else:
                    block = FIELDS[rng.randrange(len(FIELDS))]
                    if kernel == "gibbs":
                        if state.token_ids:
                            raise ValueError("Finite text Gibbs is a separate model from token generation")
                        values = finite_domains[block]
                        if len(set(values)) != len(values) or state.fields[block] not in values:
                            raise ValueError("Invalid complete Gibbs domain")
                        candidates = [state.replace(block, value) for value in values]
                        candidate = candidates[categorical(rng, softmax([score(z) for z in candidates]))]
                        forward = reverse = 0.
                    else:
                        candidate = model.sample_field(task, state, block, kernel, rng)
                        forward = float(model.log_field(task, state, candidate, block, kernel))
                        reverse = float(model.log_field(task, candidate, state, block, kernel))
                        counters["proposal_evaluations"] += 2
                proposed = score(candidate)
                log_alpha = 0. if kernel == "gibbs" else mh_log_accept(current, proposed, forward, reverse)
                accepted = math.log(max(rng.random(), 1e-300)) < log_alpha
                if accepted:
                    state, current = candidate, proposed
                    counters[kernel + "/accepted"] += 1
            else:
                counters["lazy"] += 1
            trace.append({"chain": chain_id, "transition": step, "kernel": kernel,
                          "accepted": accepted, "log_alpha": log_alpha if log_alpha != -math.inf else None,
                          "state": state.key(), "log_joint": current})
            # Rejected and lazy states are posterior samples too.
            if step >= config.burn_in and (step - config.burn_in + 1) % config.thin == 0:
                kept.append(state)
        chains.append(kept)
    flat = [z for chain in chains for z in chain]
    counts = Counter(z.key() for z in flat)
    unique = {z.key(): z for z in flat}
    samples = [PosteriorSample(unique[k], n / len(flat)) for k, n in counts.items()]
    diagnostics = dict(counters)
    diagnostics.update({"retained_states": len(flat), "unique_states": len(unique),
                        "chain_unique_states": [len({z.key() for z in chain}) for chain in chains],
                        "chain_final_states": [chain[-1].to_dict() for chain in chains]})
    diagnostics.update(chain_diagnostics([[score(z) for z in chain] for chain in chains]))
    return samples, diagnostics, trace


def importance_posterior(model, task, count, rng):
    if count < 1:
        raise ValueError("Need at least one proposal")
    states = [model.sample_prior(task.public(), rng) for _ in range(count)]
    # Prior proposals cancel the prior in the importance ratio.
    weights = softmax([float(model.log_execution(task, z)) for z in states])
    return [PosteriorSample(z, w) for z, w in zip(states, weights)], weight_diagnostics(weights)
