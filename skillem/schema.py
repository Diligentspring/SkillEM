from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FIELDS = ("when_to_apply", "preconditions", "steps", "verification", "recovery")
BENCHMARKS = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Skill:
    """A latent token state; IDs preserve sampling/scoring identity across decoding."""

    fields: dict[str, str]
    token_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if tuple(self.fields) != FIELDS or not all(isinstance(v, str) for v in self.fields.values()):
            raise ValueError(f"Skill fields must occur in fixed order {FIELDS}")
        if self.token_ids and set(self.token_ids) != set(FIELDS):
            raise ValueError("A token skill must contain IDs for every field")

    def key(self) -> str:
        return fingerprint(self.to_dict())

    def text(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.fields.items())

    def replace(self, name: str, value: str, ids: tuple[int, ...] | None = None) -> Skill:
        if name not in FIELDS:
            raise ValueError(name)
        fields, tokens = dict(self.fields), dict(self.token_ids)
        fields[name] = value
        if tokens:
            if ids is None:
                raise ValueError("Replacing a token field requires its exact IDs")
            tokens[name] = tuple(ids)
        return Skill(fields, tokens)

    def to_dict(self) -> dict:
        return {"fields": self.fields, "token_ids": {k: list(v) for k, v in self.token_ids.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        return cls({k: d["fields"][k] for k in FIELDS},
                   {k: tuple(v) for k, v in d.get("token_ids", {}).items()})


@dataclass
class Step:
    action: str
    observation: str | None = None
    # Includes the sampled terminal token when present. Teacher data can be re-tokenized.
    token_ids: list[int] | None = None


@dataclass
class Task:
    task_id: str
    question: str
    answers: list[str]
    data_source: str
    split: str
    steps: list[Step] = field(default_factory=list)
    verified: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id or not self.question.strip() or not self.answers:
            raise ValueError("Task needs an ID, a question, and answer aliases")
        if not all(isinstance(a, str) for a in self.answers):
            raise ValueError("Answer aliases must be strings")

    def public(self) -> Task:
        """The deployment prior never receives gold answers or a success trajectory."""
        return Task(self.task_id, self.question, ["<withheld>"], self.data_source, self.split)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(**{**d, "steps": [Step(**s) for s in d.get("steps", [])]})


def read_tasks(path: str | Path) -> list[Task]:
    # JSONL uses ASCII newlines. str.splitlines() also splits U+2028/U+0085
    # inside valid JSON strings, which occur in the released QA questions.
    with Path(path).open(encoding="utf-8") as handle:
        return [Task.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_json(path: str | Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_tasks(path: str | Path, tasks: list[Task]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(t.to_dict(), ensure_ascii=False) + "\n" for t in tasks))
    tmp.replace(path)
