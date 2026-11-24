"""SkillRL search wire protocol, prompt rendering, and official EM scorer."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import defaultdict
from dataclasses import replace

from .schema import Skill, Step, Task
from .vendor.search_metrics import compute_score, extract_solution
from .vendor.search_projection import search_projection
from .vendor.search_prompts import SEARCH_TEMPLATE_NO_HIS, SEARCH_TEMPLATE, SEARCH_TEMPLATE_WITH_MEMORY


def execution_prompt(task: Task, skill: Skill | None, history: list[Step], history_length=4,
                     inject_initial=True):
    projected, _ = search_projection([s.action for s in history])
    history_text = "\n".join(
        f"Step {i + 1}:{projected[i]} {s.observation or ''}\n"
        for i, s in enumerate(history)
        if i >= len(history) - history_length
    ) if history_length else ""
    kwargs = dict(task_description=task.question, memory_context=history_text, step_count=len(history))
    if skill is not None and (history or inject_initial):
        # SkillEM must condition the first action on z. Upstream omits memory at reset.
        return SEARCH_TEMPLATE_WITH_MEMORY.format(retrieved_memories=skill.text(), **kwargs)
    return (SEARCH_TEMPLATE if history and history_length else SEARCH_TEMPLATE_NO_HIS).format(**kwargs)


def official_score(steps: list[Step], answers: list[str]):
    # Upstream scores concatenated assistant actions, excluding tool observations.
    actions, _ = search_projection([s.action for s in steps])
    return float(compute_score("".join(actions), {"target": answers}))


def format_retrieval(payload):
    results = payload.get("result", [])
    if not results:
        return json.dumps({"result": "No search results found."})
    pretty = []
    for retrieval in results:
        pretty.append("".join(f"Doc {i + 1}: {item['document']['contents'].strip()}\n"
                              for i, item in enumerate(retrieval)))
    return json.dumps({"result": "\n---\n".join(pretty)})


class SearchClient:
    def __init__(self, url="http://127.0.0.1:8030/retrieve", topk=3, timeout=60):
        self.url, self.topk, self.timeout = url, topk, timeout

    def search(self, query):
        request = urllib.request.Request(
            self.url, data=json.dumps({"query": query.strip(), "topk": self.topk,
                                       "return_scores": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        # Infrastructure errors fail explicitly instead of becoming training demonstrations.
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return format_retrieval(json.load(response))


def rollout(model, task: Task, skill: Skill | None, client, rng, max_steps=4, temperature=1.):
    if max_steps < 1:
        raise ValueError("Need a positive environment step budget")
    steps, start, invalid, calls = [], time.monotonic(), 0, 0
    for _ in range(max_steps):
        action, ids = model.generate_action(task.public(), skill, steps, rng, temperature)
        # Same projection priority as SkillRL: search takes precedence if both tags occur.
        projected_actions, valids = search_projection([action])
        projected = projected_actions[0]
        search = re.search(r"<search>(.*?)</search>", projected, re.DOTALL)
        answer = re.search(r"<answer>(.*?)</answer>", projected, re.DOTALL)
        valid = bool(valids[0])
        invalid += int(not valid)
        # The upstream projection strips reasoning before passing an action to the env.
        done = len(steps) + 1 >= max_steps or (answer is not None and search is None)
        observation = None
        if not done and search is not None:
            observation = f"<information>{client.search(search.group(1))}</information>"
            calls += 1
        elif not done:
            # Upstream empty/no-query search produces an empty observation.
            observation = ""
        # Preserve raw model output as y, projected actions as the next env history.
        steps.append(Step(action, observation, ids))
        if done:
            break
    success = official_score(steps, task.answers) == 1.
    result = replace(task, steps=steps, verified=success, metadata={**task.metadata,
                     "search_calls": calls, "invalid_actions": invalid,
                     "elapsed_seconds": time.monotonic() - start,
                     "generated_tokens": sum(len(s.token_ids or []) for s in steps),
                     "termination": "answer" if answer and not search else "max_steps",
                     "skill": skill.to_dict() if skill else None})
    return result


def evaluate(tasks: list[Task]):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task.data_source].append(task)
    summary = {}
    for name, rows in {**grouped, "all": tasks}.items():
        if not rows:
            continue
        summary[name] = {"count": len(rows), "em": sum(official_score(t.steps, t.answers) for t in rows) / len(rows),
                         "mean_steps": sum(len(t.steps) for t in rows) / len(rows),
                         "mean_search_calls": sum(t.metadata.get("search_calls", 0) for t in rows) / len(rows),
                         "mean_generated_tokens": sum(t.metadata.get("generated_tokens", 0) for t in rows) / len(rows)}
    if grouped:
        summary["macro_em"] = sum(summary[k]["em"] for k in grouped) / len(grouped)
    return summary
