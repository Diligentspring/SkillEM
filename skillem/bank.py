"""A dependency-free lexical bank baseline adapted from SearchLM-ExSkill.

This ranking is an explicit lexical baseline, not SkillRL template retrieval.
The runtime accepts both task_specific_skills and query_type_skills.
"""
import json
import re
from pathlib import Path
from .schema import Skill


class StaticSkillBank:
    def __init__(self, path, top_k=6):
        self.path, self.top_k = str(path), top_k
        bank = json.loads(Path(path).read_text())
        self.records = list(bank.get("general_skills", []))
        categories = bank.get("task_specific_skills")
        if categories is None:
            categories = bank.get("query_type_skills", {})
        for group in categories.values():
            self.records.extend(group)
        self.mistakes = bank.get("common_mistakes", [])
        if not self.records:
            raise ValueError("Empty SkillBank")

    def retrieve(self, question):
        words = set(re.findall(r"\w+", question.lower()))
        def score(skill):
            return len(words & set(re.findall(r"\w+", json.dumps(skill).lower())))
        records = sorted(self.records, key=score, reverse=True)[:self.top_k]
        return Skill({"when_to_apply": "\n".join(s.get("when_to_apply", "") for s in records),
                      "preconditions": "Check the entities and requested relation.",
                      "steps": "\n".join(s.get("principle", "") for s in records),
                      "verification": "Check the answer against retrieved evidence.",
                      "recovery": "\n".join(s.get("how_to_avoid", "") for s in self.mistakes[:3])})
