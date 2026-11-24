"""Import an existing Hugging Face hub snapshot without duplicating its files."""

import argparse
import json
import os
from pathlib import Path


def import_snapshot(
    cache: Path, store: Path, repo_id: str, revision: str, *, tokenizer: bool = False
) -> Path:
    if len(repo_id.split("/")) != 2 or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError("Expected a Hugging Face owner/repository ID")
    repository = cache / ("models--" + repo_id.replace("/", "--"))
    source = repository / "snapshots" / revision
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = store / ("tokenizers" if tokenizer else "models") / repo_id
    destination.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.stat().st_dev:
        raise ValueError("Cache and LanLLM store must be on the same filesystem for hardlinks")
    for entry in source.rglob("*"):
        if entry.is_dir():
            continue
        resolved = entry.resolve(strict=True)
        if not resolved.is_relative_to(repository.resolve()):
            raise ValueError(f"Snapshot file points outside its cache repository: {entry}")
        target = destination / entry.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.samefile(resolved):
                continue
            raise FileExistsError(f"Refusing to replace {target}")
        os.link(resolved, target)
    if not tokenizer:
        if not (destination / "config.json").is_file():
            raise ValueError(f"No config.json in {source}")
        marker = destination / ".lanllm-ready"
        temporary = destination / f".lanllm-ready.{os.getpid()}"
        temporary.write_text(
            json.dumps({"revision": "main", "source_revision": revision}) + "\n",
            encoding="utf-8",
        )
        temporary.replace(marker)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("revision")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tokenizer", action="store_true")
    args = parser.parse_args()
    print(import_snapshot(args.cache, args.store, args.repo_id, args.revision, tokenizer=args.tokenizer))


if __name__ == "__main__":
    main()
