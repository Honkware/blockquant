from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_health_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "celery" in response.json()


def test_submit_job_builds_config(monkeypatch):
    captured = {}

    def delay(config):
        captured["config"] = config
        return SimpleNamespace(id="job-123")

    monkeypatch.setattr(api_main, "CELERY_OK", True)
    monkeypatch.setattr(api_main, "run_quantization", SimpleNamespace(delay=delay))
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    response = client.post(
        "/api/v1/quant",
        json={
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "format": "exl3",
            "variants": ["4.0"],
            "provider": "runpod",
            "hf_org": "blockblockblock",
            "runpod_cloud_type": "SECURE",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "status": "queued",
        "check_url": "/api/v1/jobs/job-123",
    }
    assert captured["config"]["format"] == "exl3"
    assert captured["config"]["provider"] == "runpod"
    assert captured["config"]["runpod_cloud_type"] == "SECURE"
    assert captured["config"]["hf_token"] == "hf-test-token"


def test_submit_job_returns_503_when_celery_is_missing(monkeypatch):
    monkeypatch.setattr(api_main, "CELERY_OK", False)

    response = client.post(
        "/api/v1/quant",
        json={"model_id": "Qwen/Qwen2.5-7B-Instruct", "variants": ["4.0"]},
    )

    assert response.status_code == 503


def test_submit_job_rejects_bad_payloads(monkeypatch):
    monkeypatch.setattr(api_main, "CELERY_OK", True)

    response = client.post(
        "/api/v1/quant",
        json={"model_id": "../bad", "provider": "lambda", "variants": ["../../oops"]},
    )

    assert response.status_code == 422


def test_job_status_maps_started_state(monkeypatch):
    class FakeResult:
        status = "STARTED"
        info = {"stage": "quantize", "percent": 42, "message": "measuring"}

        def ready(self):
            return False

    def async_result(job_id, app=None):
        assert job_id == "job-123"
        return FakeResult()

    import celery.result

    monkeypatch.setattr(api_main, "CELERY_OK", True)
    monkeypatch.setattr(celery.result, "AsyncResult", async_result)

    response = client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["progress"] == {
        "stage": "quantize",
        "percent": 42,
        "message": "measuring",
    }


def test_job_status_maps_success_state(monkeypatch):
    class FakeResult:
        status = "SUCCESS"
        info = None
        result = {"status": "complete"}

        def ready(self):
            return True

        def successful(self):
            return True

    import celery.result

    monkeypatch.setattr(api_main, "CELERY_OK", True)
    monkeypatch.setattr(celery.result, "AsyncResult", lambda job_id, app=None: FakeResult())

    response = client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert response.json()["result"] == {"status": "complete"}
