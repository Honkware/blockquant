"""Prompt container and JSONL loading (SPEC §3).

Prompt sets ship as packaged JSONL under ``exliberate/data/`` (owned by the
pipeline agent); this module only defines the container and the loaders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    """A single chat prompt. ``system`` is optional."""

    user: str
    system: str | None = None


def load_prompts(path: Path | str, limit: int | None = None) -> list[Prompt]:
    """Load prompts from a JSONL file.

    Schema per line: ``{"text": "..."}`` with optional ``"system": "..."``.
    Blank lines are ignored. ``limit`` truncates to the first ``limit`` items.
    """
    path = Path(path)
    prompts: list[Prompt] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(Prompt(user=record["text"], system=record.get("system")))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def packaged(name: str) -> Path:
    """Return the path to a packaged prompt set: ``exliberate/data/<name>.jsonl``.

    ``name`` may be given with or without the ``.jsonl`` suffix.
    """
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    resource = resources.files("exliberate").joinpath("data", name)
    return Path(str(resource))
