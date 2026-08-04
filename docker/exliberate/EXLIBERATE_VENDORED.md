# Vendored exliberate

Source copy of exliberate 0.2.0 plus the fixes found bringing it up on real
hardware. Vendored rather than installed from a registry because those fixes
are not in any published release, and the abliteration path depends on them:

- chat-template fallback for ExLlamaV3's tokenizer (without it, post-quant
  validation silently scores the two models on differently formatted prompts)
- `--limit` reaching the default command (bounds search time)
- whitening in torch on the model's device, cached across k

See `PATCHES.md` in the upstream project directory for the full writeup.
Refresh by re-copying `project/exliberate`, `pyproject.toml` and `README.md`.
