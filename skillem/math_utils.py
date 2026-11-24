import math
import random


def logsumexp(values):
    values = list(values)
    if not values or any(math.isnan(v) or v == math.inf for v in values):
        raise ValueError("Expected nonempty log probabilities without NaN/+inf")
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def softmax(values):
    values = list(values)
    z = logsumexp(values)
    if z == -math.inf:
        raise ValueError("Distribution has zero mass")
    return [math.exp(v - z) for v in values]


def categorical(rng: random.Random, probabilities):
    probabilities = list(probabilities)
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("Invalid probability vector")
    if not math.isclose(sum(probabilities), 1., abs_tol=1e-8):
        raise ValueError("Probabilities must sum to one")
    u, cumulative = rng.random(), 0.
    for i, p in enumerate(probabilities):
        cumulative += p
        if u < cumulative:
            return i
    return len(probabilities) - 1


def loo_advantages(rewards):
    if len(rewards) < 2:
        raise ValueError("Leave-one-out needs at least two independent samples")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def weight_diagnostics(weights):
    return {"ess": 1 / sum(w * w for w in weights),
            "entropy": -sum(w * math.log(w) for w in weights if w > 0)}


def chain_diagnostics(chains):
    """Scalar log-joint diagnostics; short/constant chains cannot certify mixing."""
    n = min(map(len, chains))
    if n < 4:
        return {"log_joint_rhat": None, "log_joint_ess": None,
                "diagnostic_note": "Need >=4 retained states per chain; no convergence certificate"}
    means = [sum(c) / len(c) for c in chains]
    variances = [sum((x-m)**2 for x in c)/(len(c)-1) for c, m in zip(chains,means)]
    within = sum(variances) / len(chains)
    grand = sum(means) / len(means)
    between = n * sum((m-grand)**2 for m in means)/(len(chains)-1) if len(chains)>1 else None
    rhat = math.sqrt(((n-1)/n*within+between/n)/within) if within>0 and between is not None else None
    estimates = []
    for c, mean, variance in zip(chains,means,variances):
        if variance == 0:
            estimates.append(None)
            continue
        positive_rhos = []
        for lag in range(1,min(len(c)//2,200)):
            rho = sum((c[i]-mean)*(c[i+lag]-mean) for i in range(len(c)-lag))/((len(c)-lag)*variance)
            if rho <= 0:
                break
            positive_rhos.append(rho)
        estimates.append(len(c)/(1+2*sum(positive_rhos)))
    return {"log_joint_rhat": rhat,
            "log_joint_ess": sum(estimates) if all(x is not None for x in estimates) else None,
            "diagnostic_note": "Scalar unsplit R-hat and truncated positive autocorrelation ESS; not a convergence certificate"}
