"""Tests Phase A additions to servers.py:
- global exception handler emits structured 500 with error_class + traceback_tail
- CUDA OOM RuntimeError is translated to 503 with retriable=True
- /health exposes the cached MODEL_META
- /v1/sam3/config exposes processor + tokenizer constants and DetectRequest defaults
"""
import threading

import pytest
from fastapi.testclient import TestClient

import servers


@pytest.fixture
def stub_models(monkeypatch):
    """Install lightweight fakes so endpoint code paths run without GPU/weights."""

    class _FakeBatched(dict):
        def to(self, *_): return self
        def get(self, k, d=None):
            if k == "original_sizes":
                class _T:
                    def tolist(self): return [[8, 8]]
                return _T()
            return d

    class _FakeProc:
        class _ImgProc:
            size = {"height": 1008, "width": 1008}
        image_processor = _ImgProc()

        class _Tok:
            model_max_length = 32
        tokenizer = _Tok()

        def __call__(self, **kw):
            return _FakeBatched()

        def post_process_instance_segmentation(self, *a, **kw):
            return [{"masks": None, "scores": None}]

    monkeypatch.setattr(servers, "SAM_PROCESSOR", _FakeProc())
    monkeypatch.setattr(servers, "TRK_MODEL", object())
    monkeypatch.setattr(servers, "TRK_PROCESSOR", object())
    monkeypatch.setattr(servers, "VID_MODEL", object())
    monkeypatch.setattr(servers, "VID_PROCESSOR", object())
    monkeypatch.setattr(servers, "MODEL_META", {
        "model_path": "/fake/path",
        "weights_sha256": "deadbeef" * 8,
        "sam3_image_class": "Sam3Model",
        "sam3_tracker_class": "Sam3TrackerModel",
        "sam3_video_class": "Sam3VideoModel",
        "dtype": "torch.bfloat16",
        "transformers_version": "fake",
        "torch_version": "fake",
        "cuda_capability": [8, 9],
        "vision_encoder_shared": True,
    })


def test_health_returns_expanded_model_meta(stub_models, monkeypatch):
    monkeypatch.setattr(servers, "SAM_MODEL", object())
    client = TestClient(servers.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"sam3_image": True, "sam3_tracker": True, "sam3_video": True}
    model = body["model"]
    assert model["sam3_image_class"] == "Sam3Model"
    assert model["weights_sha256"].startswith("deadbeef")
    assert model["vision_encoder_shared"] is True
    assert "transformers_version" in model


def test_sam3_config_exposes_processor_and_defaults(stub_models, monkeypatch):
    monkeypatch.setattr(servers, "SAM_MODEL", object())
    client = TestClient(servers.app)
    r = client.get("/v1/sam3/config")
    assert r.status_code == 200
    body = r.json()
    assert body["detect"]["field_names"] == [
        "image_b64", "prompt", "conf_threshold", "mask_threshold", "return_masks"
    ]
    assert body["detect"]["defaults"]["conf_threshold"] == 0.45
    assert body["detect"]["defaults"]["mask_threshold"] == 0.5
    assert body["detect"]["prompt_is_singular_string"] is True
    assert body["detect"]["no_hardcoded_min_score"] is True
    assert body["processor"]["image_processor_size"] == {"height": 1008, "width": 1008}
    assert body["processor"]["tokenizer_model_max_length"] == 32


def test_sam3_config_503_when_processor_not_loaded(monkeypatch):
    monkeypatch.setattr(servers, "SAM_PROCESSOR", None)
    client = TestClient(servers.app)
    r = client.get("/v1/sam3/config")
    assert r.status_code == 503


def test_openapi_schema_is_exposed():
    client = TestClient(servers.app)
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    detect_post = spec["paths"]["/v1/sam3/detect"]["post"]
    assert "requestBody" in detect_post
    schema_ref = detect_post["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert "DetectRequest" in schema_ref


def test_unhandled_exception_returns_structured_500(stub_models, monkeypatch):
    """Synthetic RuntimeError inside model forward should surface as 500 with
    error_class + traceback_tail, not an unstructured 'Internal Server Error'."""

    class _BoomModel:
        def __call__(self, **kw):
            raise RuntimeError("synthetic-failure-marker")

    monkeypatch.setattr(servers, "SAM_MODEL", _BoomModel())
    monkeypatch.setattr(servers, "_decode_b64_image", lambda b: _StubImg())

    client = TestClient(servers.app, raise_server_exceptions=False)
    r = client.post(
        "/v1/sam3/detect",
        json={"image_b64": "", "prompt": "blade"},
    )
    assert r.status_code == 500
    body = r.json()
    assert body["error_class"] == "RuntimeError"
    assert "synthetic-failure-marker" in body["detail"]
    assert "traceback_tail" in body and "synthetic-failure-marker" in body["traceback_tail"]


def test_cuda_oom_runtime_error_translated_to_503(stub_models, monkeypatch):
    """RuntimeError with 'out of memory' in message → 503 cuda_oom (retriable)."""

    class _OomModel:
        def __call__(self, **kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")

    monkeypatch.setattr(servers, "SAM_MODEL", _OomModel())
    monkeypatch.setattr(servers, "_decode_b64_image", lambda b: _StubImg())

    client = TestClient(servers.app, raise_server_exceptions=False)
    r = client.post(
        "/v1/sam3/detect",
        json={"image_b64": "", "prompt": "blade"},
    )
    assert r.status_code == 503
    body = r.json()
    assert body["error_class"] == "RuntimeError"
    assert body["retriable"] is True
    assert body["detail"].startswith("cuda_oom:")


def test_http_exception_still_passes_through(stub_models, monkeypatch):
    """400/503 from HTTPException must NOT be caught and rewritten by the global handler."""
    monkeypatch.setattr(servers, "SAM_MODEL", object())
    client = TestClient(servers.app)
    r = client.post("/v1/sam3/detect", json={"image_b64": "x", "prompt": "   "})
    assert r.status_code == 400  # not 500
    body = r.json()
    assert "prompt is required" in body["detail"]


def test_is_cuda_oom_helper():
    assert servers._is_cuda_oom(RuntimeError("CUDA out of memory. Tried..."))
    assert not servers._is_cuda_oom(RuntimeError("shape mismatch"))
    assert not servers._is_cuda_oom(ValueError("nope"))


def test_detect_handles_bfloat16_scores_and_masks(stub_models, monkeypatch):
    """Regression: numpy has no bfloat16 dtype. raw_scores.cpu().numpy() raised
    `TypeError: Got unsupported ScalarType BFloat16` on every WTBD request whose
    scores survived the threshold. detect() must cast to float32 first."""
    import torch

    class _BfModel:
        def __call__(self, **kw):
            return object()

    class _BfProc:
        class _ImgProc:
            size = {"height": 1008, "width": 1008}
        image_processor = _ImgProc()

        class _Tok:
            model_max_length = 32
        tokenizer = _Tok()

        def __call__(self, **kw):
            class _Batched(dict):
                def to(self, *_): return self
                def get(self, k, d=None):
                    if k == "original_sizes":
                        class _T:
                            def tolist(self): return [[8, 8]]
                        return _T()
                    return d
            return _Batched()

        def post_process_instance_segmentation(self, *a, **kw):
            # Two regions: one fake bool mask, one bfloat16 score per region.
            mask = torch.tensor([[[1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bool)
            scores = torch.tensor([0.83], dtype=torch.bfloat16)
            return [{"masks": mask, "scores": scores}]

    monkeypatch.setattr(servers, "SAM_MODEL", _BfModel())
    monkeypatch.setattr(servers, "SAM_PROCESSOR", _BfProc())
    monkeypatch.setattr(servers, "_decode_b64_image", lambda b: _StubImg())

    client = TestClient(servers.app, raise_server_exceptions=False)
    r = client.post(
        "/v1/sam3/detect",
        json={"image_b64": "", "prompt": "blade", "conf_threshold": 0.05, "return_masks": False},
    )
    # Pre-fix: 500 with "Got unsupported ScalarType BFloat16". Post-fix: 200 with one region.
    assert r.status_code == 200, f"bfloat16 path crashed: {r.text}"
    body = r.json()
    assert len(body["regions"]) == 1
    assert body["regions"][0]["score"] == pytest.approx(0.83, abs=0.02)


class _StubImg:
    size = (8, 8)
