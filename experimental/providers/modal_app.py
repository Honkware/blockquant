"""Modal app definition — EXL3-only GPU function.

No local code mounts. The function installs everything it needs inside
Modal's container image and runs the pipeline end-to-end.

Deploy:
    modal deploy blockquant/providers/modal_app.py
"""
import os

import modal

# ------------------------------------------------------------------
# Image
# ------------------------------------------------------------------
_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .run_commands(
        "apt-get update && apt-get install -y git",
        "pip install --upgrade numpy==2.1.0 pydantic==2.11.0 torch==2.6.0 "
        "safetensors>=0.4.0 huggingface-hub>=0.21.0 requests>=2.31 "
        "tenacity>=8.0 psutil>=5.9 tqdm>=4.65 transformers>=4.40 "
        "sentencepiece>=0.2.0 hf-xet>=0.1.0",
        # ExLlamaV3 is pure Python; install without deps
        "pip install exllamav3 --no-deps",
        # Download calibration data files not included in the PyPI wheel
        "python3 -c \"import urllib.request, os; d='/opt/conda/lib/python3.11/site-packages/exllamav3/conversion/standard_cal_data'; base='https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/conversion/standard_cal_data'; [urllib.request.urlretrieve(f'{base}/{f}', os.path.join(d, f)) for f in ['c4.utf8', 'code.utf8', 'multilingual.utf8', 'technical.utf8', 'tiny.utf8', 'wiki.utf8']]\"",
        # flash-attn cannot be installed in this conda-based image (C++ ABI mismatch).
        # Work around by (1) creating a stub flash_attn package so imports succeed,
        # and (2) patching exllamav3 to use PyTorch native SDPA instead of flash-attn.
        # Also patch 'flash_attn' (not just 'flash_attn_nc') since model_ls.py hardcodes it
        "python3 -c \"import pathlib; d=pathlib.Path('/opt/conda/lib/python3.11/site-packages/exllamav3'); [p.write_text(p.read_text().replace('\"attn_mode\": \"flash_attn\"', '\"attn_mode\": \"sdpa_nc\"')) for p in d.rglob('*.py') if '\"attn_mode\": \"flash_attn\"' in p.read_text()]\"",
        # Also patch .get() fallback defaults - attn.py and sliding_attn.py use these
        "python3 -c \"import pathlib; d=pathlib.Path('/opt/conda/lib/python3.11/site-packages/exllamav3'); [p.write_text(p.read_text().replace('.get(\"attn_mode\", \"flash_attn_nc\")', '.get(\"attn_mode\", \"sdpa_nc\")')) for p in d.rglob('*.py') if '.get(\"attn_mode\", \"flash_attn_nc\")' in p.read_text()]\",",
        "python3 -c \"import base64; exec(base64.b64decode('CmltcG9ydCBwYXRobGliCgpzaXRlX3BrZyA9IHBhdGhsaWIuUGF0aCgiL29wdC9jb25kYS9saWIvcHl0aG9uMy4xMS9zaXRlLXBhY2thZ2VzIikKCiMgMS4gQ3JlYXRlIHN0dWIgZmxhc2hfYXR0biBwYWNrYWdlIHNvIGFsbCBpbXBvcnRzIHN1Y2NlZWQKZmxhc2hfYXR0bl9kaXIgPSBzaXRlX3BrZyAvICJmbGFzaF9hdHRuIgpmbGFzaF9hdHRuX2Rpci5ta2RpcihleGlzdF9vaz1UcnVlKQooZmxhc2hfYXR0bl9kaXIgLyAiX19pbml0X18ucHkiKS53cml0ZV90ZXh0KCIiIgpkZWYgZmxhc2hfYXR0bl9mdW5jKCphcmdzLCAqKmt3YXJncyk6CiAgICByYWlzZSBSdW50aW1lRXJyb3IoImZsYXNoX2F0dG5fZnVuYyBjYWxsZWQgYnV0IGZsYXNoLWF0dG4gaXMgbm90IGluc3RhbGxlZCIpCgpkZWYgZmxhc2hfYXR0bl93aXRoX2t2Y2FjaGUoKmFyZ3MsICoqa3dhcmdzKToKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiZmxhc2hfYXR0bl93aXRoX2t2Y2FjaGUgY2FsbGVkIGJ1dCBmbGFzaC1hdHRuIGlzIG5vdCBpbnN0YWxsZWQiKQoKZGVmIGZsYXNoX2F0dG5fdmFybGVuX2Z1bmMoKmFyZ3MsICoqa3dhcmdzKToKICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigiZmxhc2hfYXR0bl92YXJsZW5fZnVuYyBjYWxsZWQgYnV0IGZsYXNoLWF0dG4gaXMgbm90IGluc3RhbGxlZCIpCiIiIikKcHJpbnQoZiJDcmVhdGVkIHN0dWIgZmxhc2hfYXR0biBhdCB7Zmxhc2hfYXR0bl9kaXJ9IikKCiMgMi4gQ3JlYXRlIHN0dWIga2JuZiBwYWNrYWdlCmtibmZfZGlyID0gc2l0ZV9wa2cgLyAia2JuZiIKa2JuZl9kaXIubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKKGtibmZfZGlyIC8gIl9faW5pdF9fLnB5Iikud3JpdGVfdGV4dCgiIiIKY2xhc3MgVm9jYWJ1bGFyeToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCAqYXJncywgKiprd2FyZ3MpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcigia2JuZi5Wb2NhYnVsYXJ5IGluc3RhbnRpYXRlZCBidXQga2JuZiBpcyBub3QgaW5zdGFsbGVkIikKCmNsYXNzIFRva2VuOgogICAgZGVmIF9faW5pdF9fKHNlbGYsICphcmdzLCAqKmt3YXJncyk6CiAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKCJrYm5mLlRva2VuIGluc3RhbnRpYXRlZCBidXQga2JuZiBpcyBub3QgaW5zdGFsbGVkIikKIiIiKQpwcmludChmIkNyZWF0ZWQgc3R1YiBrYm5mIGF0IHtrYm5mX2Rpcn0iKQoKIyAzLiBDcmVhdGUgc3R1YiBmb3JtYXRyb24gcGFja2FnZQpmb3JtYXRyb25fZGlyID0gc2l0ZV9wa2cgLyAiZm9ybWF0cm9uIgpmb3JtYXRyb25fZGlyLm1rZGlyKGV4aXN0X29rPVRydWUpCihmb3JtYXRyb25fZGlyIC8gIl9faW5pdF9fLnB5Iikud3JpdGVfdGV4dCgiIikKaW50ZWdyYXRpb25zX2RpciA9IGZvcm1hdHJvbl9kaXIgLyAiaW50ZWdyYXRpb25zIgppbnRlZ3JhdGlvbnNfZGlyLm1rZGlyKGV4aXN0X29rPVRydWUpCihpbnRlZ3JhdGlvbnNfZGlyIC8gIl9faW5pdF9fLnB5Iikud3JpdGVfdGV4dCgiIikKdXRpbHNfZGlyID0gaW50ZWdyYXRpb25zX2RpciAvICJ1dGlscyIKdXRpbHNfZGlyLm1rZGlyKGV4aXN0X29rPVRydWUpCih1dGlsc19kaXIgLyAiX19pbml0X18ucHkiKS53cml0ZV90ZXh0KCIiIgpkZWYgZ2V0X29yaWdpbmFsX2NoYXJhY3RlcnMoKmFyZ3MsICoqa3dhcmdzKTogcGFzcwpkZWYgZGVmYXVsdF9tYXNrX2xvZ2l0c19mbigqYXJncywgKiprd2FyZ3MpOiBwYXNzCmRlZiBnZXRfYml0X21hc2soKmFyZ3MsICoqa3dhcmdzKTogcGFzcwoiIiIpCmZvcm1hdHRlcl9kaXIgPSBmb3JtYXRyb25fZGlyIC8gImZvcm1hdHRlciIKZm9ybWF0dGVyX2Rpci5ta2RpcihleGlzdF9vaz1UcnVlKQooZm9ybWF0dGVyX2RpciAvICJfX2luaXRfXy5weSIpLndyaXRlX3RleHQoIiIiCmNsYXNzIEZvcm1hdHRlckJ1aWxkZXI6CiAgICBkZWYgYnVpbGQoc2VsZiwgKmFyZ3MsICoqa3dhcmdzKToKICAgICAgICByYWlzZSBSdW50aW1lRXJyb3IoIkZvcm1hdHRlckJ1aWxkZXIgdXNlZCBidXQgZm9ybWF0cm9uIGlzIG5vdCBpbnN0YWxsZWQiKQoiIiIpCmNvbmZpZ19kaXIgPSBmb3JtYXRyb25fZGlyIC8gImNvbmZpZyIKY29uZmlnX2Rpci5ta2RpcihleGlzdF9vaz1UcnVlKQooY29uZmlnX2RpciAvICJfX2luaXRfXy5weSIpLndyaXRlX3RleHQoIiIiCmNsYXNzIEVuZ2luZUdlbmVyYXRpb25Db25maWc6CiAgICBwYXNzCiIiIikKcHJpbnQoZiJDcmVhdGVkIHN0dWIgZm9ybWF0cm9uIGF0IHtmb3JtYXRyb25fZGlyfSIpCgojIDQuIFBhdGNoIGFsbCBleGxsYW1hdjMgZmlsZXMgdG8gdXNlIHNkcGFfbmMgaW5zdGVhZCBvZiBmbGFzaF9hdHRuX25jCmV4bGxhbWFfZGlyID0gc2l0ZV9wa2cgLyAiZXhsbGFtYXYzIgpmb3IgcHlfZmlsZSBpbiBleGxsYW1hX2Rpci5yZ2xvYigiKi5weSIpOgogICAgdGV4dCA9IHB5X2ZpbGUucmVhZF90ZXh0KCkKICAgIG5ld190ZXh0ID0gdGV4dC5yZXBsYWNlKCciYXR0bl9tb2RlIjogImZsYXNoX2F0dG5fbmMiJywgJyJhdHRuX21vZGUiOiAic2RwYV9uYyInKQogICAgaWYgbmV3X3RleHQgIT0gdGV4dDoKICAgICAgICBweV9maWxlLndyaXRlX3RleHQobmV3X3RleHQpCiAgICAgICAgcHJpbnQoZiJQYXRjaGVkIHtweV9maWxlLnJlbGF0aXZlX3RvKHNpdGVfcGtnKX0iKQoKIyA1LiBGaXggcXdlbjNfNS5weSB0byBub3QgcmVxdWlyZSBwcmVwcm9jZXNzb3JfY29uZmlnLmpzb24gZm9yIE1vRSBtb2RlbHMKcXdlbjM1X3B5ID0gZXhsbGFtYV9kaXIgLyAiYXJjaGl0ZWN0dXJlIiAvICJxd2VuM181LnB5IgppZiBxd2VuMzVfcHkuZXhpc3RzKCk6CiAgICB0ZXh0ID0gcXdlbjM1X3B5LnJlYWRfdGV4dCgpCiAgICAjIFBhdGNoIGJvdGggUXdlbjNfNVZMQmFzZUNvbmZpZyBhbmQgUXdlbjNfNVZMTW9lQmFzZUNvbmZpZyB2aXNpb24gYmxvY2tzCiAgICBvbGRfYmxvY2sgPSAiIiIgICAgICAgICAgICBwcmVwX3BhdGggPSBvcy5wYXRoLmpvaW4oc2VsZi5kaXJlY3RvcnksICJwcmVwcm9jZXNzb3JfY29uZmlnLmpzb24iKQogICAgICAgICAgICB3aXRoIG9wZW4ocHJlcF9wYXRoLCBlbmNvZGluZyA9ICJ1dGY4IikgYXMgZjoKICAgICAgICAgICAgICAgIHJlYWRfcHJlcF9jb25maWcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgc2VsZi52aXNpb25fcHAgPSByZWFkX3F3ZW4zX3ZsX3BwX2NvbmZpZyhyZWFkX3ByZXBfY29uZmlnKSIiIgogICAgbmV3X2Jsb2NrID0gIiIiICAgICAgICAgICAgcHJlcF9wYXRoID0gb3MucGF0aC5qb2luKHNlbGYuZGlyZWN0b3J5LCAicHJlcHJvY2Vzc29yX2NvbmZpZy5qc29uIikKICAgICAgICAgICAgaWYgb3MucGF0aC5leGlzdHMocHJlcF9wYXRoKToKICAgICAgICAgICAgICAgIHdpdGggb3BlbihwcmVwX3BhdGgsIGVuY29kaW5nID0gInV0ZjgiKSBhcyBmOgogICAgICAgICAgICAgICAgICAgIHJlYWRfcHJlcF9jb25maWcgPSBqc29uLmxvYWQoZikKICAgICAgICAgICAgICAgIHNlbGYudmlzaW9uX3BwID0gcmVhZF9xd2VuM192bF9wcF9jb25maWcocmVhZF9wcmVwX2NvbmZpZykKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIHNlbGYudmlzaW9uX3BwID0gTm9uZSIiIgogICAgbmV3X3RleHQgPSB0ZXh0LnJlcGxhY2Uob2xkX2Jsb2NrLCBuZXdfYmxvY2spCiAgICBpZiBuZXdfdGV4dCAhPSB0ZXh0OgogICAgICAgIHF3ZW4zNV9weS53cml0ZV90ZXh0KG5ld190ZXh0KQogICAgICAgIHByaW50KGYiUGF0Y2hlZCB7cXdlbjM1X3B5LnJlbGF0aXZlX3RvKHNpdGVfcGtnKX0gKHByZXByb2Nlc3Nvcl9jb25maWcuanNvbikiKQo=').decode())\"",
    )
)

app = modal.App("blockquant", image=_image)


# ------------------------------------------------------------------
# GPU function
# ------------------------------------------------------------------

@app.function(
    gpu="A100-80GB",
    timeout=43200,
)
def _run_pipeline_remote(config_dict: dict) -> dict:
    """Self-contained EXL3 pipeline run on Modal GPU."""
    import os
    import sys
    import time
    import traceback
    from pathlib import Path

    from huggingface_hub import snapshot_download, HfApi, login as hf_login

    t0 = time.time()
    model_id = config_dict["model_id"]
    variants = config_dict.get("variants", ["4.0"])
    hf_token = config_dict.get("hf_token", "")
    hf_org = config_dict.get("hf_org", "")
    head_bits = config_dict.get("head_bits", 8)

    workspace = Path("/tmp/blockquant-work") / model_id.replace("/", "--")
    workspace.mkdir(parents=True, exist_ok=True)

    outputs = []

    try:
        # ── Stage 1: Download ──────────────────────────────────────────
        model_dir = workspace / "model"
        try:
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                hf_login(token=hf_token)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                token=hf_token or None,
            )
            # Some Qwen3.5 MoE models have vision_config but no preprocessor_config.json.
            # ExLlamaV3 requires it, so create a dummy one for text-only quant.
            prep_path = model_dir / "preprocessor_config.json"
            if not prep_path.exists():
                import json
                prep_path.write_text(
                    json.dumps({
                        "size": {"shortest_edge": 56, "longest_edge": 56},
                        "patch_size": 14,
                        "temporal_patch_size": 2,
                        "merge_size": 2,
                        "image_mean": [0.48145466, 0.4578275, 0.40821073],
                        "image_std": [0.26862954, 0.26130258, 0.27577711],
                        "image_processor_type": "Qwen2VLImageProcessorFast",
                    })
                )
        except Exception as e:
            return {
                "status": "failed",
                "outputs": [],
                "error": f"Download failed: {e}",
                "total_time": time.time() - t0,
            }

        # ── Stage 2: Quantize ──────────────────────────────────────────
        # Runtime safety patch: ensure no flash_attn fallback defaults remain
        import pathlib as _pl, importlib.util as _iu
        _spec = _iu.find_spec("exllamav3")
        _exl3_dir = _pl.Path(_spec.origin).parent if _spec and _spec.origin else _pl.Path("/opt/conda/lib/python3.11/site-packages/exllamav3")
        for _p in _exl3_dir.rglob("*.py"):
            _txt = _p.read_text()
            if '.get("attn_mode", "flash_attn_nc")' in _txt:
                _p.write_text(_txt.replace('.get("attn_mode", "flash_attn_nc")', '.get("attn_mode", "sdpa_nc")'))
                print(f"[runtime patch] {_p.relative_to(_exl3_dir)}")
            if '"attn_mode": "flash_attn"' in _txt:
                _p.write_text(_txt.replace('"attn_mode": "flash_attn"', '"attn_mode": "sdpa_nc"'))
                print(f"[runtime patch] {_p.relative_to(_exl3_dir)}")
            if '"attn_mode": "flash_attn_nc"' in _txt:
                _p.write_text(_txt.replace('"attn_mode": "flash_attn_nc"', '"attn_mode": "sdpa_nc"'))
                print(f"[runtime patch] {_p.relative_to(_exl3_dir)}")

        from exllamav3.conversion.convert_model import parser, main, prepare

        for variant in variants:
            bpw = float(variant)
            out_dir = workspace / f"output-{bpw}bpw"
            work_dir = workspace / f"work-{bpw}"

            if (out_dir / "config.json").exists():
                outputs.append({"variant": variant, "path": str(out_dir)})
                continue

            old_argv = sys.argv
            sys.argv = [
                "convert",
                "-i", str(model_dir),
                "-o", str(out_dir),
                "-w", str(work_dir),
                "-b", str(bpw),
                "--head_bits", str(head_bits),
                "--parallel_mode",
            ]
            try:
                args = parser.parse_args()
            finally:
                sys.argv = old_argv

            _in_args, _job_state, _ok, _err = prepare(args)
            if not _ok:
                return {
                    "status": "failed",
                    "outputs": outputs,
                    "error": f"ExLlamaV3 prepare failed for {bpw}bpw: {_err}",
                    "total_time": time.time() - t0,
                }

            try:
                main(_in_args, _job_state)
            except Exception as e:
                return {
                    "status": "failed",
                    "outputs": outputs,
                    "error": f"ExLlamaV3 failed for {bpw}bpw: {type(e).__name__}: {e}",
                    "total_time": time.time() - t0,
                }
            outputs.append({"variant": variant, "path": str(out_dir)})

        # ── Stage 3: Upload ────────────────────────────────────────────
        if hf_token:
            api = HfApi(token=hf_token)
            model_name = model_id.split("/")[-1]
            repo_id = f"{hf_org}/{model_name}-exl3" if hf_org else f"{model_name}-exl3"
            for out in outputs:
                out_path = Path(out["path"])
                api.upload_folder(
                    folder_path=str(out_path),
                    repo_id=repo_id,
                    path_in_repo=out["variant"],
                    repo_type="model",
                )

        return {
            "status": "complete",
            "outputs": outputs,
            "error": None,
            "total_time": time.time() - t0,
        }

    except Exception as e:
        return {
            "status": "failed",
            "outputs": outputs,
            "error": (
                f"Unhandled exception: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            ),
            "total_time": time.time() - t0,
        }
