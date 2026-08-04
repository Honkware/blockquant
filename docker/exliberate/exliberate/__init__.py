"""exliberate — fully automatic refusal removal for open-weight LLMs.

Heavy dependencies (torch, transformers, peft, sklearn) are imported lazily by
the submodules that need them, so importing :mod:`exliberate` itself is cheap
and works even in environments where only the lightweight deps are installed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
