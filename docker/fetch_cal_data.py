"""Fetch exllamav3's bundled calibration corpus into the installed package.

The exllamav3 PyPI wheel ships without conversion/standard_cal_data/*.utf8, so
convert.py fails with FileNotFoundError on c4.utf8 at "Preparing input state".
This downloads the data files from the source repo into the installed package.

Idempotent (skips files already present). Pass a target dir as argv[1] for
testing; otherwise the package's conversion/standard_cal_data dir is used.
"""
import importlib.util
import os
import sys
import urllib.request

FILES = (
    "c4.utf8", "code.utf8", "multilingual.utf8",
    "technical.utf8", "tiny.utf8", "wiki.utf8",
)
BASE = (
    "https://raw.githubusercontent.com/turboderp-org/exllamav3/master/"
    "exllamav3/conversion/standard_cal_data/"
)


def _target_dir() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    spec = importlib.util.find_spec("exllamav3")
    if not spec or not spec.origin:
        return ""
    return os.path.join(os.path.dirname(spec.origin), "conversion", "standard_cal_data")


def main() -> int:
    d = _target_dir()
    if not d:
        print("exllamav3 not found", file=sys.stderr)
        return 1
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "a").close()

    fetched = []
    for name in FILES:
        dst = os.path.join(d, name)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            urllib.request.urlretrieve(BASE + name, dst)
        except Exception as exc:
            print(f"failed to fetch {name}: {exc}", file=sys.stderr)
            return 1
        fetched.append(name)
    print(f"cal data ready in {d} (fetched: {', '.join(fetched) or 'all already present'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
