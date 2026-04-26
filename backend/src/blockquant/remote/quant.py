#!/usr/bin/env python3
"""Remote quantization entrypoint — runs inside the RunPod pod.

Reads ``/root/bq-config.json``, downloads the source weights from
HuggingFace, runs ExLlamaV3's ``convert.py`` against the requested
bits-per-weight, optionally uploads the results back to HuggingFace,
then writes a final summary to ``/root/bq-result.json`` for the local
poller to pick up.

This file is shipped two ways:
  1. Baked into the prebuilt Docker image at ``/opt/blockquant/quant.py``
     (production path — see ``docker/Dockerfile.runpod``).
  2. SFTP'd to ``/root/quant.py`` by ``RunPodProvider.run_pipeline`` for
     pods running a non-baked base image (fallback path).

The two paths produce identical behaviour. Tests live in
``backend/tests/remote/test_quant.py``.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

CONFIG_PATH = "/root/bq-config.json"
RESULT_PATH = "/root/bq-result.json"


def emit_result(payload: dict) -> None:
    """Write the final result JSON for the local poller."""
    try:
        with open(RESULT_PATH, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[fatal] could not write result: {e}", flush=True)


def _self_terminate(pod_id: str, api_key: str) -> None:
    """Best-effort RunPod self-terminate via the GraphQL API.

    Uses urllib so we don't depend on the runpod SDK being installed
    inside the pod. Errors are logged but never raised — the work is
    already done, we just want to stop burning credit.
    """
    import urllib.request
    body = json.dumps({
        "query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[self-terminate] http {r.status}", flush=True)
    except Exception as e:
        print(f"[self-terminate] WARN {type(e).__name__}: {e}",
              flush=True)


def _qwen2vl_preprocessor_shim(model_dir: Path) -> None:
    """Drop a Qwen2VL preprocessor stub if missing — required by some VL
    builds even when we're only using the LM, otherwise convert.py barfs
    when it tries to read the processor config.
    """
    prep = model_dir / "preprocessor_config.json"
    if prep.exists():
        return
    prep.write_text(json.dumps({
        "size": {"shortest_edge": 56, "longest_edge": 56},
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }))


def main() -> int:
    try:
        cfg = json.loads(Path(CONFIG_PATH).read_text())
        model_id: str = cfg["model_id"]
        variants: list[str] = cfg["variants"]
        hf_token: str = cfg.get("hf_token", "")
        hf_org: str = cfg.get("hf_org", "")
        head_bits: int = int(cfg.get("head_bits", 8))
        # Calibration tunables — fewer rows trades quality for speed.
        # ExLlamaV3 defaults are 250 rows × 2048 cols when unset.
        cal_rows: int | None = cfg.get("cal_rows")
        cal_cols: int | None = cfg.get("cal_cols")

        t0 = time.time()

        import torch
        print(
            f"[gpu] CUDA: {torch.cuda.is_available()} | "
            f"{torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

        from huggingface_hub import HfApi, snapshot_download, login as hf_login

        workspace = Path("/workspace/blockquant")
        workspace.mkdir(parents=True, exist_ok=True)
        model_dir = workspace / "model"

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            hf_login(token=hf_token)

        print(f"[download] {model_id} ...", flush=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            token=hf_token or None,
        )
        print("[download] complete", flush=True)

        _qwen2vl_preprocessor_shim(model_dir)

        from exllamav3.conversion.convert_model import parser, main as exl_main, prepare

        outputs = []
        for variant in variants:
            bpw = float(variant)
            out_dir = workspace / f"output-{bpw}bpw"
            work_dir = workspace / f"work-{bpw}"
            if (out_dir / "config.json").exists():
                print(f"[skip] {variant} exists at {out_dir}", flush=True)
                outputs.append({"variant": variant, "path": str(out_dir)})
                continue
            print(f"[quantize] {variant} bpw ...", flush=True)
            old_argv = sys.argv
            argv = [
                "convert",
                "-i", str(model_dir),
                "-o", str(out_dir),
                "-w", str(work_dir),
                "-b", str(bpw),
                "--head_bits", str(head_bits),
                "--parallel_mode",
            ]
            if cal_rows is not None:
                argv += ["--cal_rows", str(int(cal_rows))]
            if cal_cols is not None:
                argv += ["--cal_cols", str(int(cal_cols))]
            sys.argv = argv
            try:
                args = parser.parse_args()
            finally:
                sys.argv = old_argv
            in_args, job_state, ok, err = prepare(args)
            if not ok:
                emit_result({"status": "failed", "error": f"prepare failed: {err}"})
                return 1
            exl_main(in_args, job_state)
            print(f"[quantize] {variant} complete", flush=True)
            outputs.append({"variant": variant, "path": str(out_dir)})

        if hf_token:
            print("[upload] to HuggingFace ...", flush=True)
            api = HfApi(token=hf_token)
            model_name = model_id.split("/")[-1]
            # Resolve the user portion when no org was supplied — HF rejects
            # bare slugs without a namespace.
            owner = hf_org or api.whoami()["name"]
            for out in outputs:
                slug = f"{model_name}-exl3-{out['variant']}bpw"
                repo_id = f"{owner}/{slug}"
                print(f"[upload] {out['variant']} -> {repo_id} ...", flush=True)
                api.create_repo(
                    repo_id=repo_id, repo_type="model",
                    exist_ok=True, private=False,
                )
                api.upload_folder(
                    folder_path=out["path"],
                    repo_id=repo_id,
                    repo_type="model",
                )
                out["hf_repo_id"] = repo_id
                out["hf_revision"] = "main"
                out["hf_url"] = f"https://huggingface.co/{repo_id}"
                # Echo the URL so the dashboard's HF_URL parser rule can
                # populate state["hf_url"] — without this, the run's last
                # log line is just "[upload] complete" with no URL.
                print(f"[upload] {out['variant']} done -> {out['hf_url']}", flush=True)
            print("[upload] complete", flush=True)

        emit_result({
            "status": "complete",
            "outputs": outputs,
            "total_time": time.time() - t0,
        })
        print("[done]", flush=True)

        # Self-terminate so we don't burn credit waiting for the local
        # poll loop to clean up. Skip on --keep-pod (forensics path).
        # Only fires on the success path; failure path below leaves the
        # pod alive for rescue_upload.py / debugging.
        if not cfg.get("keep_pod"):
            pid = cfg.get("pod_id", "")
            key = cfg.get("runpod_api_key", "")
            if pid and key:
                # Brief grace so the local poll loop has a chance to
                # scoop bq-result.json before SSH dies. Default poll
                # interval is 30 s, so this matches one tick.
                time.sleep(30)
                print(f"[self-terminate] terminating pod {pid} ...",
                      flush=True)
                _self_terminate(pid, key)
            else:
                print("[self-terminate] skipped (missing pod_id or api_key)",
                      flush=True)
        else:
            print("[self-terminate] skipped (--keep-pod)", flush=True)
        return 0

    except Exception as e:
        traceback.print_exc()
        emit_result({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
