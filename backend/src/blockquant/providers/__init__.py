"""Provider registry.

Only providers that have a validated end-to-end path are registered here.
Experimental / untested providers live under ``experimental/providers/``
until they can be re-validated — see ``experimental/README.md``.
"""
from blockquant.providers.base import Provider
from blockquant.providers.local import LocalProvider

PROVIDERS = {
    "local": LocalProvider,
}


def get_provider(name: str, **kwargs) -> Provider:
    """Instantiate a provider by name.

    Args:
        name: Provider name (currently "local" or "runpod").
        **kwargs: Passed to the provider constructor.

    Raises:
        ValueError: If the provider name is unknown or shelved.
    """
    cls = PROVIDERS.get(name)
    if cls:
        return cls(**kwargs)
    if name == "runpod":
        # Lazy-loaded — has heavyweight SDK (paramiko, runpod) we don't
        # want to import when only the local path is used.
        from blockquant.providers.runpod_provider import RunPodProvider
        return RunPodProvider(**kwargs)
    shelved = {"modal", "lambda", "vast"}
    if name in shelved:
        raise ValueError(
            f"Provider '{name}' is shelved under experimental/ and is not "
            f"currently supported. See experimental/README.md."
        )
    raise ValueError(
        f"Unknown provider: {name!r}. "
        f"Available: {', '.join(list(PROVIDERS.keys()) + ['runpod'])}"
    )
