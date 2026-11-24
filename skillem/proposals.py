"""Normalized, scoreable proposals for independent importance sampling."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .math_utils import logsumexp
from .schema import Skill


def scalar_logprob(value):
    # Keep this module usable by the dependency-free finite audit backend.
    if hasattr(value, "detach"):
        value = value.detach()
    value = float(value)
    if math.isnan(value) or value == math.inf or value > 1e-8:
        raise ValueError("Expected a normalized sequence log probability, without NaN/+inf")
    return value


@dataclass(frozen=True)
class ISConfig:
    samples: int = 16
    proposal: str = "prior"
    prior_ratio: float = .25
    deduplicate: bool = False
    proposal_epochs: int = 0

    def validate(self):
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("IS samples must be a positive integer")
        if self.proposal not in {"prior", "mixture"}:
            raise ValueError("IS proposal must be prior or mixture")
        if not isinstance(self.prior_ratio, (int, float)) or not math.isfinite(self.prior_ratio):
            raise ValueError("IS prior_ratio must be finite")
        if self.proposal == "mixture" and not 0 < self.prior_ratio < 1:
            raise ValueError("Mixture prior_ratio must be in (0,1); use prior for a pure prior proposal")
        if type(self.deduplicate) is not bool:
            raise ValueError("IS deduplicate must be boolean")
        if type(self.proposal_epochs) is not int or self.proposal_epochs < 0:
            raise ValueError("IS proposal_epochs must be a nonnegative integer")
        if self.proposal_epochs and self.proposal != "mixture":
            raise ValueError("Proposal fitting requires a mixture inference model")


def is_config(config):
    """SNIS remains a prior-only alias, including its historical default K=4."""
    if config.get("method") == "snis":
        if config.get("is_em"):
            raise ValueError("snis is prior-only; use method=is_em for proposal options")
        result = ISConfig(samples=config.get("samples", 4))
    else:
        options = config.get("is_em", {})
        if not isinstance(options, dict):
            raise ValueError("is_em must be a configuration object")
        try:
            result = ISConfig(**{"samples": config.get("samples", 16), **options})
        except TypeError as exc:
            raise ValueError(f"Invalid IS configuration: {exc}") from exc
    result.validate()
    return result


def needs_inference_model(config):
    return (config.get("method") == "variational" or
            config.get("method") == "is_em" and is_config(config).proposal == "mixture")


@dataclass(frozen=True)
class ProposalDraw:
    skill: Skill
    component: str


@dataclass(frozen=True)
class ProposalScore:
    log_prob: float
    log_prior: float
    log_inference: float | None = None


class PriorProposal:
    kind = "prior"

    def __init__(self, theta):
        self.theta = theta

    def sample(self, task, rng):
        return ProposalDraw(self.theta.sample_prior(task.public(), rng), "prior")

    def score(self, task, skill):
        prior = scalar_logprob(self.theta.log_prior(task.public(), skill))
        return ProposalScore(prior, prior)

    def log_prob(self, task, skill):
        return self.score(task, skill).log_prob


class MixtureProposal(PriorProposal):
    kind = "mixture"

    def __init__(self, theta, phi, prior_ratio=.25):
        ISConfig(proposal="mixture", prior_ratio=prior_ratio).validate()
        if phi is None:
            raise ValueError("Mixture proposal needs a scoreable inference model")
        super().__init__(theta)
        self.phi, self.prior_ratio = phi, prior_ratio

    def sample(self, task, rng):
        # Each complete skill is an IID draw from the full mixture, not a quota.
        if rng.random() < self.prior_ratio:
            return super().sample(task, rng)
        return ProposalDraw(self.phi.sample_inference(task, rng), "inference")

    def score(self, task, skill):
        prior = scalar_logprob(self.theta.log_prior(task.public(), skill))
        inference = scalar_logprob(self.phi.log_inference(task, skill))
        # Score both components even if only one of them generated this state.
        mixture = logsumexp([math.log(self.prior_ratio) + prior,
                             math.log1p(-self.prior_ratio) + inference])
        return ProposalScore(mixture, prior, inference)


def make_proposal(theta, config: ISConfig, phi=None):
    config.validate()
    return (PriorProposal(theta) if config.proposal == "prior" else
            MixtureProposal(theta, phi, config.prior_ratio))
