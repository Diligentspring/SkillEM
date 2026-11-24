#!/usr/bin/env python3
"""Download public pinned assets atomically, with checksums and retryable partials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REV = "4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, expected_sha=None, retries=4):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = dest.with_name(dest.name + ".receipt.json")
    if dest.exists() and expected_sha and sha256(dest) == expected_sha:
        record = {"url": url, "bytes": dest.stat().st_size, "sha256": expected_sha}
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return record
    if dest.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record["url"] == url and record["sha256"] == sha256(dest):
            return record
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "SkillEM-research/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            attempt_url = url
            if attempt % 2 and "/resolve/" in url and "?" not in url:
                attempt_url += "?download=true"
            request = urllib.request.Request(attempt_url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent Content-Range")
                mode = "ab" if append else "wb"
                length = response.headers.get("Content-Length")
                expected = int(length) + (offset if append else 0) if length else None
                with open(part, mode) as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if expected is not None and part.stat().st_size != expected:
                    raise IOError("Incomplete response")
            digest = sha256(part)
            if expected_sha and digest != expected_sha:
                raise ValueError("SHA256 mismatch")
            part.replace(dest)
            record = {"url": url, "bytes": dest.stat().st_size, "sha256": digest}
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"downloaded": str(dest.relative_to(ROOT)), **record}), flush=True)
            return record
        except (OSError, urllib.error.URLError) as exc:
            print(f"Download attempt {attempt + 1}/{retries}: {type(exc).__name__}: {exc}", flush=True)
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["source", "data", "model", "retrieval", "all"], default="source")
    p.add_argument("--endpoint", default="https://huggingface.co")
    p.add_argument("--max-file-gb", type=float, default=2., help="Safety budget per file; increase for GPU assets")
    p.add_argument("--metadata-only", action="store_true", help="Download configs and tokenizers, skipping model tensors")
    args = p.parse_args()
    materials = ROOT / "materials"
    if args.group in {"source", "all"}:
        archive = materials / f"SkillRL-{SOURCE_REV[:12]}.tar.gz"
        fetch(f"https://codeload.github.com/aiming-lab/SkillRL/tar.gz/{SOURCE_REV}", archive,
              "c83c7fab29b50341d5424762701257eab5db33e21a05959ece953b1030437500")
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(materials, filter="data")
            else:
                raise RuntimeError("Use Python with tarfile.data_filter for safe archive extraction")
        fetch("https://arxiv.org/pdf/2602.08234v1", materials / "skillrl_paper.pdf")
    if args.group != "source":
        lock = json.loads((ROOT / "configs/assets.lock.json").read_text())
        for asset in lock["assets"]:
            if args.group != "all" and asset["group"] != args.group:
                continue
            if args.metadata_only and (asset["file"].endswith((".safetensors", ".bin", ".gz"))
                                      or asset["file"].startswith("part_")):
                continue
            if asset.get("bytes", 0) > args.max_file_gb * 1024**3:
                print(f"Skipping budget: {asset['repo']}:{asset['file']}", flush=True)
                continue
            prefix = "datasets/" if asset["repo_type"] == "dataset" else ""
            url = f"{args.endpoint.rstrip('/')}/{prefix}{asset['repo']}/resolve/{asset['revision']}/{asset['file']}"
            fetch(url, ROOT / asset["destination"], asset.get("sha256"))


if __name__ == "__main__":
    main()
