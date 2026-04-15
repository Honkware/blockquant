#!/usr/bin/env python3
"""
Build README.md for a quantized EXL3 upload: HF model-card YAML + intro, then base repo README as reference.
Writes --output and prints one JSON line: {"ok": true} or {"ok": false, "error": "..."}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

MAX_UPSTREAM_BODY_CHARS = 120_000


def resolve_owner(api, token: str, org: str) -> str:
    if org and org.strip():
        return org.strip()
    return api.whoami(token=token)["name"]


def split_hf_readme(text: str) -> tuple[str | None, str]:
    """If README starts with YAML front matter, return (yaml, body). Else (None, full)."""
    if not text:
        return None, ""
    s = text.lstrip("\ufeff")
    if not s.startswith("---"):
        return None, s.strip()
    lines = s.splitlines()
    if lines[0].strip() != "---":
        return None, s.strip()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_part = "\n".join(lines[1:i]).strip()
            body = "\n".join(lines[i + 1 :]).strip()
            return yaml_part, body
    return None, s.strip()


def yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def fetch_upstream_readme(repo_id: str, token: str | None) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            token=token,
        )
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (EntryNotFoundError, RepositoryNotFoundError, GatedRepoError, OSError):
        return None


# Tags on the quant repo card (keep short; details live in blockquant-manifest.json on upload).
README_CARD_TAGS = [
    "exl3",
    "exllamav3",
    "quantized",
    "text-generation",
    "blockquant",
    "exllama",
]


def base_model_yaml_value(source_repo: str) -> str:
    if re.match(r"^[\w.-]+/[\w.-]+$", source_repo):
        return source_repo
    return yaml_quote(source_repo)


def bits_weight_yaml(bpw: float) -> int | float:
    if float(bpw).is_integer():
        return int(bpw)
    return bpw


def format_tags_yaml(tags: list[str]) -> str:
    return "tags:\n" + "\n".join(f"- {t}" for t in tags)


def build_readme(
    *,
    source_repo: str,
    full_repo_id: str,
    bpw: float,
    upstream_raw: str | None,
    revision: str | None = None,
) -> str:
    upstream_yaml, upstream_body = split_hf_readme(upstream_raw) if upstream_raw else (None, "")

    base_url = f"https://huggingface.co/{source_repo}"
    quant_url = f"https://huggingface.co/{full_repo_id}"

    bw = bits_weight_yaml(bpw)
    tags_block = format_tags_yaml(list(README_CARD_TAGS))

    branch_table_row = ""
    revision_bullet = ""
    if revision:
        br_url = f"{quant_url}/tree/{revision}"
        branch_table_row = f"| **HF branch** | `{revision}` ([files]({br_url})) |\n"
        revision_bullet = (
            f"- Download this branch: `huggingface-cli download {full_repo_id} --revision {revision}` "
            "(Hub API: `snapshot_download` / `hf_hub_download` with the `revision` argument).\n"
        )

    front = f"""---
base_model: {base_model_yaml_value(source_repo)}
library_name: exllamav3
{tags_block}
quantization_format: exl3
bits_per_weight: {bw}
---

# {source_repo.split("/")[-1]} — {bpw:g} bpw EXL3

This model is an [**EXL3**](https://github.com/turboderp-org/exllamav3)-quantized build of **[{source_repo}]({base_url})**, produced for GPU inference with **ExLlamaV3**.

| | |
| --- | --- |
| **Base model** | [{source_repo}]({base_url}) |
| **Format** | EXL3 (ExLlamaV3) |
| **Bits per weight** | {bw} |
| **This repo** | [{full_repo_id}]({quant_url}) |
{branch_table_row}## Inference

{revision_bullet}- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) (OpenAI-compatible API, ExLlamaV2/V3)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) with the **ExLlamaV3** loader
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) directly

## License and use

Use and license follow the **base model**. See the base repository for terms, citation, and safety documentation.

---
## Original model README (reference)

The content below is copied from the **base** repository `README.md` for convenience. **Only the YAML front matter at the very top of this file** applies to *this* model card; the sections below are reference only.

"""

    if upstream_raw is None:
        front += "\n_No `README.md` was found on the base model repo (or it is gated / unreachable with this token)._\n"
        return front

    body = upstream_body if upstream_body else upstream_raw.strip()
    if len(body) > MAX_UPSTREAM_BODY_CHARS:
        body = body[:MAX_UPSTREAM_BODY_CHARS] + "\n\n_(truncated for size; see base repo for full text.)_\n"

    if upstream_yaml:
        section_yaml = (
            "### Upstream YAML front matter (reference)\n\n```yaml\n"
            + upstream_yaml
            + "\n```\n\n"
        )
        section_body = "### Upstream README body\n\n" + body + "\n"
    else:
        section_yaml = ""
        section_body = "### Upstream README (full)\n\n" + body + "\n"

    return front + section_yaml + section_body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to write README.md")
    parser.add_argument("--source-repo", required=True, help="Base model id, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Repo name without org, e.g. Qwen3.5-4B-exl3",
    )
    parser.add_argument("--bpw", type=float, required=True)
    parser.add_argument("--org", default="")
    parser.add_argument(
        "--revision",
        default="",
        help="HF branch name for this BPW, e.g. 8.00bpw",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if token:
            owner = resolve_owner(api, token, args.org)
        else:
            owner = (args.org or "").strip() or "unknown"
        full_repo_id = f"{owner}/{args.repo_name}"

        upstream = fetch_upstream_readme(args.source_repo, token)

        rev = (args.revision or "").strip() or None
        text = build_readme(
            source_repo=args.source_repo,
            full_repo_id=full_repo_id,
            bpw=args.bpw,
            upstream_raw=upstream,
            revision=rev,
        )

        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

        print(json.dumps({"ok": True, "upstream_readme_found": upstream is not None}), flush=True)
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:500]}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
