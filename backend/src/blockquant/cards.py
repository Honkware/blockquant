"""Render the polished EXL3 model card used for every published quant.

Single source of truth for the card. The local pipeline (``stages.report``)
imports it directly; the RunPod path ships this file to the pod next to
``remote/quant.py``. It depends only on the stdlib plus ``huggingface_hub``
(present in both environments) and deliberately avoids importing the rest of
``blockquant`` so it can run standalone on a pod.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Per-bpw positioning + VRAM copy. Generic enough for any mid/large model;
# unlisted bit-widths fall back to a neutral line.
POSITIONING = {
    "3.0": "the tightest fit, sized for 16&nbsp;GB consumer cards while leaving usable context room",
    "4.0": "the tight&#8209;fit build, sized to leave generous context room on a 24&nbsp;GB consumer GPU and to load on 16&nbsp;GB cards at workable context lengths",
    "4.5": "the quality-leaning sweet spot: comfortable on a single 24&nbsp;GB consumer GPU, effectively indistinguishable from FP16 on most reasoning tasks",
    "5.0": "the quality build that fits a 24&nbsp;GB card with reduced context, with headroom on 32&nbsp;GB cards",
    "6.0": "near-lossless reference quality for 32&nbsp;GB+ cards (V100, A100, RTX&nbsp;6000)",
}

VRAM_HINT = {
    "3.0": "**VRAM at 3.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Fits a 16&nbsp;GB card with workable context, comfortable on 24&nbsp;GB with very long context.",
    "4.0": "**VRAM at 4.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Comfortable on a single 24&nbsp;GB card with room for ~24k tokens of context; fits a 16&nbsp;GB card with a ~4&ndash;6k token window.",
    "4.5": "**VRAM at 4.5&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Comfortable on a single 24&nbsp;GB card with room for ~16k tokens of context; fits a 16&nbsp;GB card with a reduced context window.",
    "5.0": "**VRAM at 5.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Tight on 24&nbsp;GB (limited context); comfortable on 32&nbsp;GB+.",
    "6.0": "**VRAM at 6.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Best on 32&nbsp;GB+ cards (V100, A100, RTX&nbsp;6000) where there's room for long context.",
}


def exl3_repo_slug(base_name: str, variant: str) -> str:
    """Canonical repo name for an EXL3 variant: ``{model}-exl3-{bpw}bpw``."""
    return f"{base_name}-exl3-{variant}bpw"


def exl3_repo_id(owner: str, base_name: str, variant: str) -> str:
    slug = exl3_repo_slug(base_name, variant)
    return f"{owner}/{slug}" if owner else slug


def _find_template() -> str:
    """Locate card_template.md across the library layout and the pod layout."""
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("BLOCKQUANT_CARD_TEMPLATE")
    if env:
        candidates.append(Path(env))
    candidates.append(here.parents[1] / "templates" / "card_template.md")  # backend/templates
    candidates.append(here / "card_template.md")  # co-located on the pod
    for path in candidates:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    raise FileNotFoundError(
        "card_template.md not found (set BLOCKQUANT_CARD_TEMPLATE or ship it next to cards.py)"
    )


def pretty_title(model_name: str, override: str | None = None) -> str:
    """Human title for the card heading.

    ``override`` wins when supplied (use it for hand-curated titles). Otherwise
    the bare model name is split on separators into a middot-joined heading,
    e.g. ``Qwen3.6-35B-A3B`` -> ``Qwen3.6 · 35B · A3B``.
    """
    if override:
        return override.strip()
    parts = [p for p in re.split(r"[-_/]+", model_name) if p]
    return " · ".join(parts) if parts else model_name


def _size_tokens(model_name: str) -> str | None:
    """Pull a size descriptor like ``35B-A3B`` or ``7B`` out of the model name."""
    m = re.search(r"(\d+(?:\.\d+)?B(?:-A\d+(?:\.\d+)?B)?)", model_name)
    return m.group(1) if m else None


def derive_model_facts(config: dict, model_name: str = "") -> dict:
    """Pull architecture facts out of a HuggingFace ``config.json`` dict."""
    archs = config.get("architectures") or []
    layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
        or config.get("n_layers")
    )
    experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
    )
    is_moe = bool(experts)

    if is_moe and layers:
        arch_line = (
            f"Mixture&#8209;of&#8209;Experts &nbsp;·&nbsp; {layers} layers "
            f"&times; {experts} experts"
        )
    elif is_moe:
        arch_line = "Mixture&#8209;of&#8209;Experts"
    elif layers:
        arch_line = f"Dense &nbsp;·&nbsp; {layers} layers"
    else:
        arch_line = "Dense"

    kind = "MoE" if is_moe else "Dense"
    size = _size_tokens(model_name)
    badge_text = f"{kind}_{size}" if size else kind
    # shields.io: literal hyphens must be doubled.
    arch_badge = badge_text.replace("-", "--")

    extra_tags = ["  - mixture-of-experts"] if is_moe else []
    parallel_line = "`enabled` (MoE expert batching)" if is_moe else "`enabled`"

    return {
        "arch_line": arch_line,
        "arch_badge": arch_badge,
        "extra_tags": "\n".join(extra_tags),
        "parallel_line": parallel_line,
        "architecture": archs[0] if archs else "",
        "is_moe": is_moe,
        "layers": layers,
        "experts": experts,
    }


def _est_size_gb(bpw: float, n_params_b: float = 35.0) -> float:
    """Coarse pre-publish size estimate when a real size isn't known yet."""
    return n_params_b * bpw / 8.0 + 1.5


def build_quants_table(rows: list[dict], current_variant: str) -> str:
    """Render the Quants table.

    Each row: ``{"variant": str, "head_bits": int, "cal_rows": int,
    "size_gb": float|None, "url": str|None}``. ``size_gb`` None means
    not-yet-published (shows an estimate + "queued").
    """
    header = (
        "| BPW &nbsp; | &nbsp; Head bits &nbsp; | "
        "&nbsp; Calibration rows &nbsp; | &nbsp; Size &nbsp; | &nbsp; Status |\n"
        "| :---: | :---: | :---: | ---: | :--- |"
    )
    body = []
    for row in sorted(rows, key=lambda r: float(r["variant"])):
        v = row["variant"]
        is_current = v == current_variant
        if row.get("size_gb") is not None:
            size_str = f"{row['size_gb']:.1f}&nbsp;GB"
            status = (
                "<kbd>this repo</kbd>"
                if is_current
                else f"[link]({row['url']})"
            )
        else:
            size_str = f"<i>~{_est_size_gb(float(v)):.0f}&nbsp;GB</i>"
            status = "<kbd>this repo</kbd>" if is_current else "<sub>queued</sub>"
        if is_current:
            size_str = f"**{size_str}**"
        bpw_cell = f"**{v}**" if is_current else v
        body.append(
            f"| {bpw_cell} | {row.get('head_bits', 8)} | "
            f"{row.get('cal_rows', 250)} | {size_str} | {status} |"
        )
    return header + "\n" + "\n".join(body)


def _render(template: str, ctx: dict) -> str:
    out = template
    for key, value in ctx.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def render_exl3_card(
    *,
    base_repo: str,
    repo_id: str,
    variant: str,
    head_bits: int,
    cal_rows: int,
    size_gb: float | None,
    model_config: dict,
    quant_rows: list[dict],
    collection_url: str,
    license_id: str = "other",
    quantized_by: str,
    title_override: str | None = None,
) -> str:
    """Render the full polished card for one EXL3 variant."""
    base_name = base_repo.split("/")[-1]
    facts = derive_model_facts(model_config, base_name)
    size_str = f"{size_gb:.1f}" if size_gb is not None else f"{_est_size_gb(float(variant)):.1f}"

    ctx = {
        "LICENSE": license_id or "other",
        "BASE_REPO": base_repo,
        "BASE_BADGE": base_repo.replace("/", "%2F").replace("-", "--"),
        "QUANTIZED_BY": quantized_by,
        "EXTRA_TAGS": facts["extra_tags"],
        "TITLE": pretty_title(base_name, title_override),
        "ARCH_LINE": facts["arch_line"],
        "ARCH_BADGE": facts["arch_badge"],
        "PARALLEL_LINE": facts["parallel_line"],
        "BPW": variant,
        "SIZE_GB": size_str,
        "SIZE_GB_BADGE": size_str,
        "HEAD_BITS": str(head_bits),
        "CAL_ROWS": str(cal_rows),
        "REPO_ID": repo_id,
        "SHORT_NAME": repo_id.split("/")[-1],
        "POSITIONING": POSITIONING.get(variant, f"the {variant}&nbsp;bpw build"),
        "VRAM_HINT": VRAM_HINT.get(
            variant,
            f"**VRAM at {variant}&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead.",
        ),
        "QUANTS_TABLE": build_quants_table(quant_rows, variant),
        "COLLECTION_URL": collection_url,
    }
    return _render(_find_template(), ctx)


# --- HuggingFace helpers (network; imported lazily so the renderer stays pure) ---

def fetch_license(base_repo: str, token: str | None = None) -> str:
    """Best-effort license id from the base model's card metadata."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(base_repo)
        return (info.card_data or {}).get("license") or "other"
    except Exception:
        return "other"


def ensure_collection(
    *, owner: str, base_name: str, token: str, item_repo_ids: list[str] | None = None
) -> str:
    """Create (idempotently) a per-model EXL3 collection and add repos to it.

    Returns the collection URL, or the owner's collections page as a fallback
    if the API call fails for any reason.
    """
    fallback = f"https://huggingface.co/{owner}"
    try:
        from huggingface_hub import (
            add_collection_item,
            create_collection,
        )

        collection = create_collection(
            title=f"{base_name} EXL3",
            namespace=owner,
            description=f"EXL3 quants of {base_name}, produced by BlockQuant.",
            exists_ok=True,
            token=token,
        )
        for repo_id in item_repo_ids or []:
            try:
                add_collection_item(
                    collection_slug=collection.slug,
                    item_id=repo_id,
                    item_type="model",
                    token=token,
                    exists_ok=True,
                )
            except Exception:
                pass
        return f"https://huggingface.co/collections/{collection.slug}"
    except Exception:
        return fallback
