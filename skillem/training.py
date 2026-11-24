"""Alternating EM with fixed observations and equal task weights."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from .data import validate_training_tasks
from .inference import MCMCConfig, PosteriorSample, sample_mcmc, importance_posterior, joint
from .math_utils import loo_advantages
from .schema import FIELDS, Skill, fingerprint, write_json


def m_step(model, tasks, posterior, optimizer, epochs=1, batch_tasks=1, max_grad_norm=1., objective="joint", rng=None):
    """Average tasks, then posterior mass; never re-normalize a partial sample batch."""
    if objective not in {"joint", "execution"} or batch_tasks < 1 or epochs < 1:
        raise ValueError("Invalid M-step options")
    for task in tasks:
        samples = posterior[task.task_id]
        if (not samples or any(not math.isfinite(s.weight) or s.weight < 0 for s in samples)
                or not math.isclose(sum(s.weight for s in samples), 1., abs_tol=1e-7)):
            raise ValueError("Each task's frozen posterior weights must sum to one")
    metrics = []
    model.model.eval()  # Dropout would change the probability model used in E-step.
    for epoch in range(epochs):
        ordered = list(tasks)
        if rng:
            rng.shuffle(ordered)
        total = 0.
        for start in range(0, len(ordered), batch_tasks):
            batch = ordered[start:start + batch_tasks]
            optimizer.zero_grad(set_to_none=True)
            for task in batch:
                for sample in posterior[task.task_id]:
                    if sample.weight == 0:
                        continue
                    scale = sample.weight / len(batch)
                    if objective == "joint":
                        for lp in model.iter_skill_scores(task, sample.skill):
                            loss = -scale * lp
                            total += float(loss.detach()) * len(batch)
                            loss.backward()
                    for lp in model.iter_execution_scores(task, sample.skill):
                        loss = -scale * lp
                        total += float(loss.detach()) * len(batch)
                        loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
        metrics.append({"epoch": epoch, "sampled_negative_Q": total / len(tasks)})
    return metrics


@torch.no_grad()
def variational_samples(theta, phi, task, count, rng):
    if count < 2:
        raise ValueError("Variational E-step requires K>=2 independent skills")
    states = [phi.sample_inference(task, rng) for _ in range(count)]
    rewards = [joint(theta, task, z) - float(phi.log_inference(task, z)) for z in states]
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("Nonfinite ELBO sample")
    return states, rewards


def variational_e_step(theta, phi, tasks, optimizer, rng, count=4, epochs=1, batch_tasks=1, max_grad_norm=1.):
    if epochs < 1 or batch_tasks < 1:
        raise ValueError("Invalid variational E-step budgets")
    metrics = []
    theta.model.eval()
    phi.model.eval()
    for epoch in range(epochs):
        elbo, seen = 0., 0
        for start in range(0, len(tasks), batch_tasks):
            batch = tasks[start:start + batch_tasks]
            optimizer.zero_grad(set_to_none=True)
            for task in batch:
                states, rewards = variational_samples(theta, phi, task, count, rng)
                advantages = loo_advantages(rewards)  # Detached Python scalars; no group std normalization.
                elbo += sum(rewards) / count
                seen += 1
                for z, advantage in zip(states, advantages):
                    for lp in phi.iter_skill_scores(task, z, mode="inference"):
                        (-advantage * lp / (len(batch) * count)).backward()
            torch.nn.utils.clip_grad_norm_(phi.model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
        metrics.append({"epoch": epoch, "elbo_sample_mean_before_updates": elbo / seen,
                        "estimator": "score_function_leave_one_out"})
    return metrics


def make_optimizer(model, lr):
    params = [p for p in model.model.parameters() if p.requires_grad]
    if not params or lr <= 0:
        raise ValueError("Need trainable parameters and positive learning rate")
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.)


def train(theta, tasks, config, output, phi=None, resume=False, validation_fn=None):
    validate_training_tasks(tasks)
    if config.get("rounds", 1) < 1:
        raise ValueError("Need at least one EM round")
    method = config.get("method", "mcmc")
    if method not in {"mcmc", "variational", "snis", "prior_uniform", "sft", "static_skill"}:
        raise ValueError(f"Unknown method: {method}")
    if method == "variational" and (phi is None or phi.model is theta.model):
        raise ValueError("Variational EM needs independent theta and phi parameters")
    if phi and (theta.tokenizer.get_vocab() != phi.tokenizer.get_vocab()
                or theta.eos != phi.eos or theta.max_field_tokens != phi.max_field_tokens):
        raise ValueError("Theta and phi must use the same latent token support")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    config_hash = fingerprint(config)
    data_hash = fingerprint([t.to_dict() for t in tasks])
    rng = random.Random(config.get("seed", 0))
    optimizer = make_optimizer(theta, config.get("learning_rate", 1e-6))
    q_optimizer = make_optimizer(phi, config.get("inference_learning_rate", 1e-6)) if phi else None
    completed, initial, history = 0, {}, []
    best_score, stale = -math.inf, 0
    if (root / "latest.json").exists():
        if not resume:
            raise ValueError("Output already contains a run; pass --resume or choose another directory")
        latest = json.loads((root / "latest.json").read_text())
        if latest["config_hash"] != config_hash or latest["data_hash"] != data_hash:
            raise ValueError("Resume requires the same resolved config and fixed training corpus")
        # Checkpoints are local files produced by this program; never load untrusted pickles.
        state = torch.load(root / latest["round_dir"] / "state.pt", map_location="cpu", weights_only=False)
        expected_theta = str((root / latest["round_dir"] / "theta").resolve())
        expected_phi = str((root / latest["round_dir"] / "phi").resolve())
        if (getattr(theta, "loaded_checkpoint", None) != expected_theta
                or phi and getattr(phi, "loaded_checkpoint", None) != expected_phi):
            raise ValueError("Load latest theta/phi checkpoints before resuming")
        optimizer.load_state_dict(state["optimizer"])
        if q_optimizer:
            q_optimizer.load_state_dict(state["q_optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed, initial, history = latest["round"], state["initial"], state["history"]
        best_score, stale = state.get("best_score", -math.inf), state.get("stale", 0)
        if latest.get("early_stopped"):
            return history
    elif resume:
        raise ValueError("No completed checkpoint to resume")
    write_json(root / "config.json", config)
    for iteration in range(completed, config.get("rounds", 1)):
        started = time.monotonic()
        before_calls = dict(theta.calls) if hasattr(theta, "calls") else {}
        before_q_calls = dict(phi.calls) if phi and hasattr(phi, "calls") else {}
        round_dir = root / f"round-{iteration + 1:03d}"
        round_dir.mkdir(exist_ok=True)
        posterior, diagnostics, traces = {}, {}, {}
        if method == "variational":
            e_metrics = variational_e_step(theta, phi, tasks, q_optimizer, rng,
                count=config.get("samples", 4), epochs=config.get("e_epochs", 1),
                batch_tasks=config.get("batch_tasks", 1))
            diagnostics["variational"] = e_metrics
        with torch.no_grad():
            for task in tasks:
                if method == "mcmc":
                    mcmc_config = MCMCConfig(**config.get("mcmc", {}))
                    starts = [Skill.from_dict(d) for d in initial[task.task_id]] if task.task_id in initial else None
                    samples, diag, trace = sample_mcmc(theta, task, mcmc_config, rng, starts)
                    diagnostics[task.task_id], traces[task.task_id] = diag, trace
                    initial[task.task_id] = diag["chain_final_states"]
                elif method == "snis":
                    samples, diag = importance_posterior(theta, task, config.get("samples", 4), rng)
                    diagnostics[task.task_id] = diag
                elif method in {"sft", "static_skill"}:
                    static = task.metadata.get("static_skill") if method == "static_skill" else None
                    if method == "static_skill" and static is None:
                        raise ValueError("static_skill baseline needs metadata.static_skill for every task")
                    skill = Skill({name: static[name] for name in FIELDS}) if static else None
                    samples = [PosteriorSample(skill, 1.)]
                else:
                    n = config.get("samples", 4)
                    if n < 1:
                        raise ValueError("Need positive sample count")
                    # After phi updates, M-step samples are freshly drawn from the new phi.
                    samples = [PosteriorSample(phi.sample_inference(task, rng) if method == "variational"
                                               else theta.sample_prior(task.public(), rng), 1 / n)
                               for _ in range(n)]
                posterior[task.task_id] = samples
        write_json(round_dir / "posterior.json", {k: [{"skill": s.skill.to_dict() if s.skill else None,
                   "weight": s.weight} for s in v] for k, v in posterior.items()})
        write_json(round_dir / "diagnostics.json", diagnostics)
        write_json(round_dir / "trace.json", traces)
        metrics = m_step(theta, tasks, posterior, optimizer, epochs=config.get("m_epochs", 1),
                         batch_tasks=config.get("batch_tasks", 1), rng=rng,
                         objective="execution" if method in {"sft", "static_skill"} else "joint")
        result = {"round": iteration + 1, "m_step": metrics}
        if validation_fn is not None:
            with torch.no_grad():
                result["validation"] = validation_fn(theta, iteration + 1)
            score = result["validation"]["macro_em"]
            if score > best_score:
                best_score, stale = score, 0
                best_round = round_dir.name
            else:
                stale += 1
                best_round = None
        else:
            best_round = None
        result["elapsed_seconds"] = time.monotonic() - started
        result["model_calls"] = {k: v - before_calls.get(k, 0) for k, v in getattr(theta, "calls", {}).items()}
        if phi:
            result["inference_model_calls"] = {k: v - before_q_calls.get(k, 0) for k, v in phi.calls.items()}
        result["claim"] = "approximate EM; no certified monotonicity"
        history.append(result)
        theta.save(round_dir / "theta")
        if phi:
            phi.save(round_dir / "phi")
        torch.save({"optimizer": optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict() if q_optimizer else None,
                    "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "initial": initial, "history": history, "best_score": best_score,
                    "stale": stale}, round_dir / "state.pt")
        write_json(round_dir / "metrics.json", history[-1])
        # This atomic marker is written only after all round files are complete.
        early_stop = bool(validation_fn and config.get("patience") and stale >= config["patience"])
        if best_round:
            write_json(root / "best.json", {"round": iteration + 1, "round_dir": best_round,
                       "validation_macro_em": best_score})
        write_json(root / "latest.json", {"round": iteration + 1, "round_dir": round_dir.name,
                   "config_hash": config_hash, "data_hash": data_hash, "early_stopped": early_stop})
        print(json.dumps(history[-1]), flush=True)
        if early_stop:
            break
    return history
