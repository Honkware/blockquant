"""Stage 3: Quantize to all requested variants with streaming progress & resume."""
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

# Calibration data auto-download (inspired by ezexl3)
_CAL_FILES = ["c4.utf8", "code.utf8", "multilingual.utf8", "technical.utf8", "wiki.utf8", "tiny.utf8"]
_CAL_BASE_URLS = [
    "https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/conversion/standard_cal_data",
]
_CAL_MAX_RETRIES = 3


def _download_with_retries(urls: list[str], dest: str) -> None:
    """Try each URL with retries and exponential backoff."""
    last_err = None
    for url in urls:
        for attempt in range(_CAL_MAX_RETRIES):
            try:
                urllib.request.urlretrieve(url, dest)
                return
            except Exception as e:
                last_err = e
                if attempt < _CAL_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


def _ensure_exl3_cal_data() -> None:
    """Download exllamav3's calibration data if the pip wheel omitted it."""
    try:
        import exllamav3.conversion as _conv
    except ImportError:
        logger.warning("exllamav3 not installed — skipping cal data check")
        return

    conv_dir = Path(_conv.__file__).parent
    cal_dir = conv_dir / "standard_cal_data"

    if (cal_dir / _CAL_FILES[0]).exists():
        return

    os.makedirs(cal_dir, exist_ok=True)
    logger.info("Downloading exllamav3 calibration data (one-time)...")
    for fname in _CAL_FILES:
        dest = cal_dir / fname
        if dest.exists():
            continue
        urls = [f"{base}/{fname}" for base in _CAL_BASE_URLS]
        try:
            _download_with_retries(urls, str(dest))
            logger.info(f"  {fname}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download calibration file {fname}: {e}\n\n"
                f"You can manually download from:\n"
                f"  {_CAL_BASE_URLS[0]}/\n\n"
                f"And place the .utf8 files in:\n"
                f"  {cal_dir}/"
            ) from e
    logger.info("Calibration data ready.")


# Regexes for parsing convert.py stdout
_MODULE_LOAD_RE = re.compile(r"-- Loading unquantized module:\s*(.+)")
_LAYER_QUANT_RE = re.compile(r"-- Quantized:\s*(\S+)")
_LAYER_UNQUANT_RE = re.compile(r"-- Unquantized:\s*(\S+)")


def _stream_subprocess_with_progress(
    cmd: list[str],
    progress_callback: Callable | None,
    stage_name: str = "quantize",
    stage_start_pct: int = 25,
    stage_end_pct: int = 85,
) -> subprocess.CompletedProcess:
    """Run a subprocess, stream stdout, and emit throttled progress events."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    modules_seen: set[str] = set()
    layers_done = 0
    current_module = ""
    last_send = 0.0

    for line in proc.stdout:
        lines.append(line)

        m = _MODULE_LOAD_RE.search(line)
        if m:
            current_module = m.group(1).strip()
            modules_seen.add(current_module)

        if _LAYER_QUANT_RE.search(line) or _LAYER_UNQUANT_RE.search(line):
            layers_done += 1

        # Throttle progress events to every 2 seconds
        now = time.time()
        if progress_callback and now - last_send >= 2.0:
            msg = f"quantizing {current_module}" if current_module else "quantizing..."
            if layers_done:
                msg += f" ({layers_done} layers processed)"
            # Coarse progress within quantize stage: 25% → 85%
            # Use modules_seen count as a rough proxy (most models have 20-40 modules)
            estimated_modules = max(len(modules_seen), 1)
            pct = min(95, int(len(modules_seen) / (estimated_modules + 5) * 100))
            overall_pct = stage_start_pct + int(pct * (stage_end_pct - stage_start_pct) / 100)
            progress_callback(stage_name, overall_pct, msg)
            last_send = now

    proc.wait()
    stdout = "".join(lines)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, "")


def run(config: QuantConfig, workspace: Path, progress_callback=None) -> dict:
    """Quantize to all variants. Returns dict with outputs or error."""
    t0 = time.time()
    outputs: list[QuantOutput] = []

    if config.format == QuantFormat.EXL3:
        _ensure_exl3_cal_data()
        for variant in config.variants:
            out_dir = workspace / f"output-{variant}bpw"
            work_dir = workspace / f"work-{variant}"

            ok = _run_exl3_quantize(
                input_dir=workspace / "model",
                output_dir=out_dir,
                work_dir=work_dir,
                bpw=float(variant),
                head_bits=config.head_bits,
                cal_rows=config.cal_rows,
                cal_cols=config.cal_cols,
                parallel_mode=config.parallel_mode,
                high_quality=variant in config.high_quality_bpws,
                head_bits_8=variant in config.head_bits_8_bpws,
                progress_callback=progress_callback,
            )
            if ok:
                size_mb = (
                    sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
                    / (1024 * 1024)
                )
                outputs.append(
                    QuantOutput(
                        variant=variant,
                        format=QuantFormat.EXL3,
                        output_path=str(out_dir),
                        file_size_mb=size_mb,
                    )
                )

    elif config.format == QuantFormat.GGUF:
        f16_path = workspace / "model.f16.gguf"
        for variant in config.variants:
            out_path = workspace / f"model-{variant}.gguf"

            ok = _run_gguf_quantize(f16_path, out_path, variant, config.use_imatrix)
            if ok:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                outputs.append(
                    QuantOutput(
                        variant=variant,
                        format=QuantFormat.GGUF,
                        output_path=str(out_path),
                        file_size_mb=size_mb,
                    )
                )

    return {
        "ok": len(outputs) > 0,
        "outputs": outputs,
        "time": time.time() - t0,
        "error": None if outputs else "All variants failed",
    }


def _run_exl3_quantize(
    input_dir,
    output_dir,
    work_dir,
    bpw,
    head_bits,
    cal_rows,
    cal_cols,
    parallel_mode: bool = False,
    high_quality: bool = False,
    head_bits_8: bool = False,
    progress_callback=None,
):
    """Run ExLlamaV3 convert.py with resume, streaming progress, and per-BPW flags."""
    exllama_dir = Path(os.environ.get("EXLLAMAV3_DIR", r"E:\BlockQuant-v2\BlockQuant\exllamav3"))
    convert_script = exllama_dir / "convert.py"
    if not convert_script.exists():
        convert_script = Path("exllamav3/convert.py")

    if not convert_script.exists():
        raise FileNotFoundError("ExLlamaV3 convert.py not found. Set EXLLAMAV3_DIR.")

    # Resume / checkpoint logic (inspired by ezexl3)
    if (output_dir / "config.json").exists():
        logger.info(f"Output already exists for {bpw}bpw — skipping")
        return True

    if (work_dir / "args.json").exists():
        logger.info(f"Resuming existing job for {bpw}bpw")
        args = [
            str(convert_script),
            "-w", str(work_dir),
            "-r",
            "-b", str(bpw),
            "--head_bits", str(head_bits),
        ]
    else:
        args = [
            str(convert_script),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-w", str(work_dir),
            "-b", str(bpw),
            "--head_bits", str(head_bits),
        ]

    if cal_rows:
        args += ["--cal_rows", str(cal_rows)]
    if cal_cols:
        args += ["--cal_cols", str(cal_cols)]
    if parallel_mode:
        args += ["--parallel_mode"]
    if high_quality:
        args += ["-hq"]
    if head_bits_8:
        args += ["-hb", "8"]

    logger.info(f"Running ExLlamaV3: bpw={bpw} {'(resuming)' if (work_dir / 'args.json').exists() else ''}")

    result = _stream_subprocess_with_progress(
        [sys.executable, "-u"] + args,
        progress_callback=progress_callback,
        stage_name="quantize",
        stage_start_pct=25,
        stage_end_pct=85,
    )

    if result.returncode != 0:
        log_file = work_dir / "exllamav3_error.log"
        log_file.write_text(result.stdout, encoding="utf-8")
        logger.error(f"ExLlamaV3 failed (see {log_file}): {result.stdout[:400]}")
        return False
    return True


def _run_gguf_quantize(f16_path: Path, out_path: Path, variant: str, use_imatrix: bool):
    """Run llama.cpp llama-quantize."""
    quant_bin = Path("llama.cpp/llama-quantize")
    if not quant_bin.exists():
        raise FileNotFoundError("llama.cpp/llama-quantize not found. Build llama.cpp first.")

    cmd = [str(quant_bin)]
    if use_imatrix:
        imatrix = f16_path.parent / "imatrix.dat"
        if imatrix.exists():
            cmd += ["--imatrix", str(imatrix)]
    cmd += [str(f16_path), str(out_path), variant]

    logger.info(f"Running llama-quantize: {variant}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log_file = out_path.parent / f"llama-quantize-{variant}-error.log"
        log_file.write_text(result.stderr, encoding="utf-8")
        logger.error(f"llama-quantize failed (see {log_file}): {result.stderr[:400]}")
        return False
    return True
