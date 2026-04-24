"""Celery tasks — these replace local subprocess calls from the Discord bot."""
from celery import Celery
from blockquant.models import QuantConfig
from blockquant.pipeline import run_pipeline

app = Celery("blockquant")
app.config_from_object("scheduler.celeryconfig")


@app.task(bind=True, max_retries=2)
def run_quantization(self, config_dict: dict):
    """Main Celery task that the Discord bot triggers."""
    try:
        config = QuantConfig(**config_dict)

        def progress_callback(stage: str, percent: int, message: str = ""):
            self.update_state(
                state="STARTED",
                meta={
                    "stage": stage,
                    "percent": percent,
                    "message": message or f"{stage} ({percent}%)",
                },
            )

        result = run_pipeline(config, progress_callback=progress_callback)
        return result.model_dump(mode="json")
    except Exception as exc:
        # Retry on OOM or spot preemption
        msg = str(exc).lower()
        if any(k in msg for k in ["oom", "out of memory", "cuda", "preempt"]):
            raise self.retry(exc=exc, countdown=60)
        raise


@app.task
def health_check():
    return {"status": "ok"}
