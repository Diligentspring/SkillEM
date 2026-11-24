from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .schema import BENCHMARKS, Task, fingerprint, write_json, write_tasks
from .search import official_score


def answers_from_row(row):
    ground = (row.get("env_kwargs") or {}).get("ground_truth")
    if ground is None:
        ground = (row.get("reward_model") or {}).get("ground_truth", row.get("golden_answers"))
    if isinstance(ground, dict):
        ground = ground.get("target", ground.get("answers"))
    if isinstance(ground, str):
        ground = [ground]
    if ground is None:
        raise ValueError("No ground truth in row")
    return list(ground)


def task_from_row(row, split, index, repository):
    extra, env = row.get("extra_info") or {}, row.get("env_kwargs") or {}
    question = row.get("question") or env.get("question") or extra.get("question")
    if question is None:
        raise ValueError("Missing question; refusing to treat a formatted prompt as the question")
    source = str(row.get("data_source", env.get("data_source", "unknown"))).lower()
    source = {"2wiki": "2wikimultihopqa", "natural_questions": "nq", "hotpot_qa": "hotpotqa"}.get(source, source)
    if source not in BENCHMARKS:
        raise ValueError(f"Unknown Search data source {source!r}")
    original_id = str(row.get("id") or extra.get("index", index))
    task_id = f"{repository}:{split}:{source}:{original_id}"
    return Task(task_id, question, answers_from_row(row), source, split,
                metadata={"repository": repository, "original_id": original_id,
                          "question_hash": fingerprint(question.strip().lower())})


def prepare_parquet(path, out_dir, repository, split, validation_fraction=.05, seed=0,
                    exclude_questions=None):
    import pyarrow.parquet as pq
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be in (0,1)")
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    relevant = [name for name in ("id", "question", "data_source", "golden_answers", "reward_model",
                                  "extra_info", "env_kwargs") if name in columns]
    rows = pq.read_table(path, columns=relevant).to_pylist()
    tasks = [task_from_row(row, split, i, repository) for i, row in enumerate(rows)]
    original_count = len(tasks)
    excluded = []
    if exclude_questions:
        if split != "train":
            raise ValueError("Exclusions apply to training only; preserve the official test set")
        exclude_questions = {q.strip().lower() for q in exclude_questions}
        excluded = [t.task_id for t in tasks if t.question.strip().lower() in exclude_questions]
        tasks = [t for t in tasks if t.question.strip().lower() not in exclude_questions]
    # Identical questions stay together; validation is drawn only from training data.
    sets = {"train": [], "validation": []} if split == "train" else {split: []}
    for task in tasks:
        if split == "train":
            key = hashlib.sha256(f"{seed}:{task.question.strip().lower()}".encode()).digest()
            target = "validation" if int.from_bytes(key[:8], "big") / 2**64 < validation_fraction else "train"
        else:
            target = split
        task.split = target
        sets[target].append(task)
    out = Path(out_dir)
    for name, group in sets.items():
        write_tasks(out / f"{name}.jsonl", group)
    write_json(out / f"{split}.manifest.json", {
        "source_file": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "repository": repository, "split": split, "seed": seed,
        "validation_fraction": validation_fraction if split == "train" else None,
        "counts": {k: len(v) for k, v in sets.items()},
        "source_rows": original_count, "excluded_overlap_ids": excluded,
        "exclusion_question_fingerprint": fingerprint(sorted(exclude_questions)) if exclude_questions else None,
        "data_sources": dict(Counter(t.data_source for t in tasks)),
        "ids": {k: [t.task_id for t in v] for k, v in sets.items()}})
    return {k: len(v) for k, v in sets.items()}


def validate_training_tasks(tasks):
    if not tasks:
        raise ValueError("Training corpus is empty")
    ids = set()
    for t in tasks:
        if t.task_id in ids:
            raise ValueError(f"Duplicate task ID {t.task_id}; group multiple demonstrations explicitly")
        ids.add(t.task_id)
        if t.split != "train" or not t.verified or not t.steps or official_score(t.steps, t.answers) != 1:
            raise ValueError(f"Expected verified training success: {t.task_id}")


def audit_splits(groups):
    seen, report = {}, {}
    for name, tasks in groups.items():
        report[name] = len(tasks)
        for task in tasks:
            question = task.question.strip().lower()
            if question in seen and seen[question] != name:
                raise ValueError(f"Question leakage between {seen[question]} and {name}: {task.task_id}")
            seen[question] = name
    return report
