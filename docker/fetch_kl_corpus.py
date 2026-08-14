"""Bake the held-out corpus the KL metric is measured on into the image.

The metric used to run on exllamav3's conversion/standard_cal_data -- the very
corpus EXL3 calibrates the quant against. Measuring KL there is measuring on
the training set: it flatters every quant and cannot be read against
turboderp's published curves, which use openwebtext precisely to avoid this.

openwebtext is also disjoint from the calibration set by construction: that is
c4, code, multilingual, technical, tiny and wiki, none of which this is.

Rows come from the HF datasets-server over plain HTTP rather than through
`datasets`, because the small openwebtext mirrors are script-based and modern
`datasets` will not execute a dataset script. No library, no script execution,
and paging by offset makes the result reproducible.

    python fetch_kl_corpus.py /opt/blockquant/kl_eval_corpus.utf8
"""
import hashlib
import json
import sys
import time
import urllib.request

DATASET = "Skylion007/openwebtext"
CONFIG = "plain_text"
SPLIT = "train"
PAGE = 100          # the rows endpoint's maximum
# 8 x 8192 tokens is ~65k tokens; at ~4 chars/token that is ~260k characters.
# Take well past it so chat formatting and a fat tokenizer cannot leave the
# eval short of a full 8 rows.
TARGET_CHARS = 2_000_000
TRIES = 5


def fetch(offset: int) -> list[str]:
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET.replace('/', '%2F')}"
           f"&config={CONFIG}&split={SPLIT}&offset={offset}&length={PAGE}")
    for i in range(TRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            return [row["row"].get("text", "") for row in data.get("rows", [])]
        except Exception as e:  # noqa: BLE001
            if i == TRIES - 1:
                raise RuntimeError(f"rows at offset {offset}: {e}") from e
            time.sleep(5 * (i + 1))
    return []


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/opt/blockquant/kl_eval_corpus.utf8"
    parts: list[str] = []
    total = 0
    offset = 0
    while total < TARGET_CHARS:
        rows = fetch(offset)
        if not rows:
            break
        for t in rows:
            t = (t or "").strip()
            if t:
                parts.append(t)
                total += len(t)
        offset += PAGE

    text = "\n\n".join(parts)
    if len(text) < TARGET_CHARS // 4:
        # Fail the build. An image that silently falls back to the calibration
        # corpus publishes a number that looks like the others and is not.
        print(f"FATAL: only {len(text)} chars of eval text", file=sys.stderr)
        return 1

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {out}: {len(parts)} docs, {len(text)} chars, sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
