"""A fully enumerable latent model for checking posterior inference and EM."""
import itertools
import math

from .inference import exact_posterior
from .math_utils import categorical
from .schema import FIELDS, Skill, Task


class FiniteModel:
    def __init__(self):
        self.domains = {name: ["0", "1"] if i < 2 else ["fixed"] for i, name in enumerate(FIELDS)}
        self.states = [Skill(dict(zip(FIELDS, values))) for values in itertools.product(*self.domains.values())]
        self.prior = [.1, .2, .3, .4]
        self.emissions = [[.9, .1], [.3, .7], [.7, .3], [.1, .9]]

    def _index(self, z):
        return next(i for i, s in enumerate(self.states) if s.fields == z.fields)

    def log_prior(self, task, z):
        return math.log(self.prior[self._index(z)])

    def log_execution(self, task, z):
        return math.log(self.emissions[self._index(z)][task.metadata["outcome"]])

    def sample_prior(self, task, rng):
        return self.states[categorical(rng, self.prior)]

    def field_prob(self, current, candidate, name, mode):
        values = self.domains[name]
        if mode == "local":
            # Deliberately asymmetric to detect a missing reverse correction.
            return [.8, .2][values.index(candidate.fields[name])] if len(values) == 2 else 1.
        return 1 / len(values)

    def sample_field(self, task, current, name, mode, rng):
        candidates = [current.replace(name, v) for v in self.domains[name]]
        probs = [self.field_prob(current, z, name, mode) for z in candidates]
        return candidates[categorical(rng, probs)]

    def log_field(self, task, current, candidate, name, mode):
        return math.log(self.field_prob(current, candidate, name, mode))

    def exact_em_update(self, tasks):
        posterior = [exact_posterior(self, t, self.states)[0] for t in tasks]
        counts = [[0., 0.] for _ in self.states]
        for t, samples in zip(tasks, posterior):
            for i, s in enumerate(samples):
                counts[i][t.metadata["outcome"]] += s.weight
        self.prior = [sum(row) / len(tasks) for row in counts]
        self.emissions = [[v / sum(row) for v in row] for row in counts]


def finite_tasks():
    return [Task(f"finite-{i}", "Finite audit observation", ["audit"], "nq", "train",
                 metadata={"outcome": y}) for i, y in enumerate([0, 0, 0, 1, 1])]
