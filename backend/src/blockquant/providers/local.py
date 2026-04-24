"""Local provider — runs pipeline directly on the current machine."""
from blockquant.providers.base import Provider


class LocalProvider(Provider):
    name = "local"

    def launch(self, config: dict) -> str:
        return "local"

    def terminate(self, instance_id: str):
        pass

    def run(self, instance_id: str, command: str) -> dict:
        # Not used for local — pipeline is called directly
        return {"stdout": "", "stderr": "", "code": 0}
