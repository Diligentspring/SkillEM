from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .schema import FIELDS, read_tasks, write_tasks, write_json, Skill


def load_config(path):
    return json.loads(Path(path).read_text())


def load_backend(config, name=None, inference=False):
    from .model import HFModel
    args = dict(config["model"])
    model_name = name or args.pop("name")
    args.pop("name", None)
    if inference:
        model_name = name or config.get("inference_model", {}).get("name", model_name)
        args.update({k: v for k, v in config.get("inference_model", {}).items() if k != "name"})
    backend = HFModel.load(model_name, **args)
    return backend


def main(argv=None):
    parser = argparse.ArgumentParser(description="SkillEM: latent skill learning for SkillRL Search QA")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--split", choices=["train", "test"], required=True)
    prepare.add_argument("--validation-fraction", type=float, default=.05)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--exclude-questions", nargs="+", help="Held-out JSONL files; exclude their question keys from train")
    audit = sub.add_parser("audit-data")
    audit.add_argument("files", nargs="+", help="JSONL split files")
    for command in ("collect", "evaluate", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--limit", type=int)
        if command == "train":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--validation", help="Held-out validation JSONL; never a test split")
        else:
            p.add_argument("--checkpoint")
            p.add_argument("--skill-mode", choices=["prior", "none", "static", "drop", "swap"], default="prior")
            p.add_argument("--skill-bank", help="SkillRL JSON bank for the static baseline")
    demo = sub.add_parser("finite-demo")
    demo.add_argument("--output", default="runs/finite-demo.json")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_parquet
        excluded = {t.question for path in (args.exclude_questions or []) for t in read_tasks(path)}
        print(json.dumps(prepare_parquet(args.input, args.output, args.repository, args.split,
                                        args.validation_fraction, args.seed, excluded)))
        return
    if args.command == "audit-data":
        from .data import audit_splits
        groups = {}
        for f in args.files:
            for task in read_tasks(f):
                groups.setdefault(task.split, []).append(task)
        print(json.dumps(audit_splits(groups)))
        return
    if args.command == "finite-demo":
        from .finite import FiniteModel, finite_tasks
        from .inference import exact_posterior, sample_mcmc, MCMCConfig
        model, tasks = FiniteModel(), finite_tasks()
        likelihoods = []
        for _ in range(5):
            likelihoods.append(sum(exact_posterior(model, t, model.states)[1] for t in tasks) / len(tasks))
            model.exact_em_update(tasks)
        samples, diagnostics, _ = sample_mcmc(model, tasks[0],
            MCMCConfig(chains=2, burn_in=200, samples_per_chain=10000,
                       kernels={"independent": .4, "local": .2, "mwg": .2, "gibbs": .2}),
            random.Random(0), finite_domains=model.domains)
        exact, _ = exact_posterior(model, tasks[0], model.states)
        estimated = {s.skill.key(): s.weight for s in samples}
        tv = .5 * sum(abs(s.weight - estimated.get(s.skill.key(), 0)) for s in exact)
        result = {"exact_em_log_likelihood": likelihoods, "mcmc_total_variation": tv,
                  "diagnostics": diagnostics, "scope": "finite probability audit; not a QA result"}
        write_json(args.output, result)
        print(json.dumps(result))
        return
    config = load_config(args.config)
    tasks = read_tasks(args.input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    if not tasks:
        parser.error("Input is empty")
    import torch
    torch.manual_seed(config.get("seed", 0))
    if args.command == "train":
        from .training import train
        theta_name = phi_name = None
        if args.resume:
            latest = load_config(Path(args.output) / "latest.json")
            theta_name = str(Path(args.output) / latest["round_dir"] / "theta")
            phi_name = str(Path(args.output) / latest["round_dir"] / "phi")
        theta = load_backend(config, theta_name)
        phi = load_backend(config, phi_name, inference=True) if config.get("method") == "variational" else None
        validation_fn = None
        if args.validation:
            from .data import audit_splits
            from .search import SearchClient, rollout, evaluate
            validation = read_tasks(args.validation)
            if not validation or any(t.split != "validation" for t in validation):
                parser.error("--validation must contain only a nonempty validation split")
            audit_splits({"train": tasks, "validation": validation})
            # Round-independent subsampling and RNG make validation budgets comparable.
            selection_rng = random.Random(config.get("seed", 0) + 1000003)
            selection_rng.shuffle(validation)
            validation = validation[:config.get("validation_limit", len(validation))]
            if not validation:
                parser.error("validation_limit must retain at least one task")
            env = dict(config.get("environment", {}))
            max_steps, temperature = env.pop("max_steps", 4), env.pop("temperature", 1.)
            client = SearchClient(**env)
            def validation_fn(model, iteration):
                rng = random.Random(config.get("seed", 0) + 2000003)
                episodes = []
                for task in validation:
                    if config.get("method") == "sft":
                        z = None
                    elif config.get("method") == "static_skill":
                        if "static_skill" not in task.metadata:
                            raise ValueError("Attach static skills to validation tasks too")
                        z = Skill({name: task.metadata["static_skill"][name] for name in FIELDS})
                    else:
                        z = model.sample_prior(task.public(), rng)
                    episodes.append(rollout(model, task, z, client, rng, max_steps, temperature))
                write_tasks(Path(args.output) / f"round-{iteration:03d}" / "validation.jsonl", episodes)
                return evaluate(episodes)
            # The validation corpus is part of the run identity for safe resume.
            from .schema import fingerprint
            config["validation_fingerprint"] = fingerprint([t.to_dict() for t in validation])
        train(theta, tasks, config, args.output, phi, args.resume, validation_fn)
    else:
        from .search import SearchClient, rollout, evaluate
        from .bank import StaticSkillBank
        model = load_backend(config, args.checkpoint)
        seed = config.get("seed", 0)
        env = dict(config.get("environment", {}))
        max_steps, temperature = env.pop("max_steps", 4), env.pop("temperature", 1.)
        client = SearchClient(**env)
        bank = StaticSkillBank(args.skill_bank) if args.skill_mode == "static" and args.skill_bank else None
        if args.skill_mode == "static" and bank is None:
            parser.error("static mode needs --skill-bank")
        with torch.no_grad():
            def skill_for(task):
                if bank:
                    return bank.retrieve(task.question)
                if args.skill_mode == "none":
                    return None
                return model.sample_prior(task.public(), random.Random(f"{seed}:skill:{task.task_id}"))
            skills = (skill_for(t) for t in tasks)
            if args.skill_mode == "swap":
                if len(tasks) < 2:
                    parser.error("swap intervention requires at least two tasks")
                skills = list(skills)
                skills = skills[1:] + skills[:1]
            results = []
            root = Path(args.output)
            root.mkdir(parents=True, exist_ok=True)
            if (root / "episodes.jsonl").exists():
                parser.error("Output already has episodes; choose a new evaluation directory")
            with (root / "episodes.jsonl").open("w") as out:
                successes = (root / "successes.jsonl").open("w") if args.command == "collect" else None
                try:
                    for task, skill in zip(tasks, skills):
                        if args.skill_mode == "drop":
                            skill = None
                        rng = random.Random(f"{seed}:execution:{task.task_id}")
                        result = rollout(model, task, skill, client, rng, max_steps, temperature)
                        results.append(result)
                        line = json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                        out.write(line)
                        out.flush()
                        if successes and result.verified:
                            successes.write(line)
                            successes.flush()
                        print(json.dumps({"completed": len(results), "task_id": task.task_id,
                                          "success": result.verified}), flush=True)
                finally:
                    if successes:
                        successes.close()
        write_json(root / "metrics.json", evaluate(results))
        write_json(root / "run.json", {"config": config, "input": str(Path(args.input).resolve()),
                   "checkpoint": args.checkpoint, "skill_mode": args.skill_mode, "calls": model.calls})
        print(json.dumps(evaluate(results)))


if __name__ == "__main__":
    main()
