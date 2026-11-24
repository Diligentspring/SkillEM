"""Hugging Face backend with matched normalized sampling and sequence scoring.

A skill is five bounded token fields. At the field length bound EOS is forced
with probability one; elsewhere non-EOS special tokens are masked and logits
are re-normalized. The prior, inference model, and proposals use this same
support. Exact sampled token IDs are serialized, so decode/re-encode cannot
silently change MH states. Executor actions use the ordinary LM distribution.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import FIELDS, Skill, Task, Step, write_json
from .search import execution_prompt


class ContextOverflow(ValueError):
    pass


class HFModel:
    def __init__(self, model, tokenizer, max_field_tokens=64, max_action_tokens=700,
                 max_context=8192, history_length=4, inject_initial=True):
        self.model, self.tokenizer = model, tokenizer
        self.max_field_tokens, self.max_action_tokens = max_field_tokens, max_action_tokens
        self.max_context, self.history_length = max_context, history_length
        self.inject_initial = inject_initial
        if tokenizer.eos_token_id is None or min(max_field_tokens, max_action_tokens) < 1:
            raise ValueError("Need an EOS token and positive generation limits")
        self.eos = tokenizer.eos_token_id
        vocab_size = model.config.vocab_size
        self.banned = [i for i in tokenizer.all_special_ids if i != self.eos and i < vocab_size]
        # Disable dropout for a deterministic probability model, including training forwards.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            for key in ("attention_dropout", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
                value = getattr(module, key, None)
                if isinstance(value, float):
                    setattr(module, key, 0.)
        for key in vars(self.model.config):
            if "dropout" in key or "pdrop" in key:
                value = getattr(self.model.config, key)
                if isinstance(value, (int, float)):
                    setattr(self.model.config, key, 0.)
        self.model.eval()
        self.calls = {"scoring_forwards": 0, "generation_forwards": 0, "generated_tokens": 0}

    @classmethod
    def load(cls, name, revision=None, device="cpu", dtype="float32", lora_rank=0,
             gradient_checkpointing=False, **kwargs):
        path = Path(name)
        checkpoint = path / "skillem_model.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            kwargs = {**saved["distribution"], **kwargs}
            if saved.get("adapter"):
                from peft import PeftModel
                tok = AutoTokenizer.from_pretrained(name)
                base = AutoModelForCausalLM.from_pretrained(saved["base_model"],
                           revision=saved.get("base_revision"), torch_dtype=getattr(torch, dtype),
                           attn_implementation="eager")
                model = PeftModel.from_pretrained(base, name, is_trainable=True)
                if gradient_checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.enable_input_require_grads()
                obj = cls(model.to(device), tok, **kwargs)
                obj.base_model, obj.base_revision = saved["base_model"], saved.get("base_revision")
                obj.loaded_checkpoint = str(path.resolve())
                return obj
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision,
                    torch_dtype=getattr(torch, dtype), attn_implementation="eager")
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank,
                      lora_dropout=0., target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM"))
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        obj = cls(model.to(device), tok, **kwargs)
        obj.base_model, obj.base_revision = str(name), revision
        obj.loaded_checkpoint = str(path.resolve()) if checkpoint.exists() else None
        return obj

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _chat(self, content):
        messages = [{"role": "system", "content": "You are a helpful and harmless assistant."},
                    {"role": "user", "content": content}]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # Explicit fallback supports tiny local audit models, not the official Qwen experiment.
        return self.tokenizer.encode("User: " + content + "\nAssistant:", add_special_tokens=True)

    def _check(self, prefix, target_len):
        if not prefix or len(prefix) + target_len > self.max_context:
            raise ContextOverflow(f"{len(prefix)} context + {target_len} target exceeds {self.max_context}; no truncation applied")

    def _normalized(self, logits, bounded):
        logits = logits.float()
        if bounded and self.banned:
            logits = logits.clone()
            logits[..., self.banned] = -torch.inf
        return torch.log_softmax(logits, dim=-1)

    def sequence_logprob(self, prefix, target, bounded=False):
        """Sum token probabilities; prompt and observations contribute no labels."""
        target = list(target)
        self._check(prefix, len(target))
        if not target:
            raise ValueError("A sequence must contain a terminal token")
        if bounded:
            if (target[-1] != self.eos or self.eos in target[:-1] or
                    len(target) > self.max_field_tokens + 1 or
                    any(t in self.banned for t in target)):
                raise ValueError("Skill field is outside the declared bounded support")
        ids = torch.tensor([prefix + target], dtype=torch.long, device=self.device)
        # Training mode enables HF gradient checkpointing. Dropout is explicitly zero.
        self.model.train(torch.is_grad_enabled())
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        self.calls["scoring_forwards"] += 1
        positions = logits[0, len(prefix)-1:len(prefix)+len(target)-1]
        lp = self._normalized(positions, bounded)
        selected = lp.gather(-1, torch.tensor(target, device=self.device)[:, None]).squeeze(-1)
        if bounded and len(target) == self.max_field_tokens + 1:
            # At the length bound EOS is deterministic, not an unnormalized truncation.
            selected = selected[:-1]
        return selected.sum()

    @torch.no_grad()
    def _sample(self, prefix, rng, bounded=False, temperature=1.):
        if temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        self.model.eval()
        limit = self.max_field_tokens if bounded else self.max_action_tokens
        self._check(prefix, limit + int(bounded))
        generator = torch.Generator(device="cpu").manual_seed(rng.randrange(2**63-1))
        target, past = [], None
        ids = torch.tensor([prefix], device=self.device)
        for _ in range(limit):
            out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
            self.calls["generation_forwards"] += 1
            logits = out.logits[0, -1] / temperature
            probabilities = self._normalized(logits, bounded).exp().cpu()
            token = int(torch.multinomial(probabilities, 1, generator=generator))
            target.append(token)
            if token == self.eos:
                break
            past = out.past_key_values
            ids = torch.tensor([[token]], device=self.device)
        if bounded and target[-1] != self.eos:
            target.append(self.eos)
        self.calls["generated_tokens"] += len(target)
        return tuple(target)

    def _field_prompt(self, task, field, previous=None, mode="prior", current=None):
        body = ("Write a concise reusable procedural skill. Do not copy instance-specific answers or evidence.\n"
                f"Question: {task.question}\nOutput only the content of field: {field}. End with EOS.\n")
        if mode != "prior":
            body += "Verified trajectory:\n" + json.dumps(
                [{"action": s.action, "observation": s.observation} for s in task.steps], ensure_ascii=False) + "\n"
        if mode in {"local", "mwg"}:
            visible = dict(current.fields)
            if mode == "mwg":
                del visible[field]  # Infilling does not read the old value.
            body += "Current skill fields:\n" + json.dumps(visible, ensure_ascii=False) + "\n"
        elif previous:
            # Include latent IDs as well as text to retain the complete prefix state.
            body += "Previous fields:\n" + json.dumps(previous, ensure_ascii=False) + "\n"
        return self._chat(body)

    def _decode(self, ids):
        return self.tokenizer.decode([i for i in ids if i != self.eos], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def _full_sample(self, task, rng, mode):
        fields, tokens, previous = {}, {}, {}
        for name in FIELDS:
            ids = self._sample(self._field_prompt(task, name, previous, mode), rng, bounded=True)
            tokens[name], fields[name] = ids, self._decode(ids)
            previous[name] = {"text": fields[name], "token_ids": list(ids)}
        return Skill(fields, tokens)

    def iter_skill_scores(self, task, skill, mode="prior"):
        self._validate_skill(skill)
        if mode == "prior":
            task = task.public()
        previous = {}
        for name in FIELDS:
            yield self.sequence_logprob(self._field_prompt(task, name, previous, mode),
                                       skill.token_ids[name], bounded=True)
            previous[name] = {"text": skill.fields[name], "token_ids": list(skill.token_ids[name])}

    def _full_score(self, task, skill, mode):
        return torch.stack(list(self.iter_skill_scores(task, skill, mode))).sum()

    def _validate_skill(self, skill):
        if not skill.token_ids:
            raise ValueError("HF latent skills require exact token IDs; use encode_skill for initialization")
        for k in FIELDS:
            if self._decode(skill.token_ids[k]) != skill.fields[k]:
                raise ValueError("Skill text differs from its latent token IDs")

    def encode_skill(self, fields):
        tokens = {}
        for name in FIELDS:
            ids = self.tokenizer.encode(fields[name], add_special_tokens=False)
            if self.eos in ids or any(i in self.banned for i in ids) or len(ids) > self.max_field_tokens:
                raise ValueError("Initial skill exceeds support; increase max_field_tokens explicitly")
            tokens[name] = tuple(ids + [self.eos])
        return Skill(dict(fields), tokens)

    def sample_prior(self, task, rng):
        return self._full_sample(task.public(), rng, "prior")

    def log_prior(self, task, skill):
        return self._full_score(task.public(), skill, "prior")

    def sample_inference(self, task, rng):
        return self._full_sample(task, rng, "inference")

    def log_inference(self, task, skill):
        return self._full_score(task, skill, "inference")

    def sample_field(self, task, current, name, mode, rng):
        ids = self._sample(self._field_prompt(task, name, mode=mode, current=current), rng, bounded=True)
        return current.replace(name, self._decode(ids), ids)

    def log_field(self, task, current, candidate, name, mode):
        self._validate_skill(candidate)
        if any(current.fields[k] != candidate.fields[k] or current.token_ids[k] != candidate.token_ids[k]
               for k in FIELDS if k != name):
            raise ValueError("A field proposal changed another block")
        return self.sequence_logprob(self._field_prompt(task, name, mode=mode, current=current),
                                    candidate.token_ids[name], bounded=True)

    def _action_prefix(self, task, skill, history):
        return self._chat(execution_prompt(task, skill, history, self.history_length, self.inject_initial))

    def iter_execution_scores(self, task, skill):
        if not task.steps:
            raise ValueError("A likelihood requires an observed trajectory")
        for i, step in enumerate(task.steps):
            target = step.token_ids
            if target is None:
                target = self.tokenizer.encode(step.action, add_special_tokens=False) + [self.eos]
            else:
                if self._decode(target) != step.action:
                    raise ValueError("Action text/token IDs do not match the loaded tokenizer")
            yield self.sequence_logprob(self._action_prefix(task, skill, task.steps[:i]), target)

    def log_execution(self, task, skill):
        return torch.stack(list(self.iter_execution_scores(task, skill))).sum()

    def generate_action(self, task, skill, history, rng, temperature=1.):
        ids = self._sample(self._action_prefix(task, skill, history), rng, temperature=temperature)
        return self._decode(ids), list(ids)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        write_json(path / "skillem_model.json", {
            "base_model": getattr(self, "base_model", None), "base_revision": getattr(self, "base_revision", None),
            "adapter": hasattr(self.model, "peft_config"),
            "distribution": {"max_field_tokens": self.max_field_tokens, "max_action_tokens": self.max_action_tokens,
                             "max_context": self.max_context, "history_length": self.history_length,
                             "inject_initial": self.inject_initial}})
