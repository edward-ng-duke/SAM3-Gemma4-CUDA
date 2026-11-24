import servers


def _reset_model_globals(monkeypatch):
    for name in (
        "SAM_MODEL",
        "SAM_PROCESSOR",
        "TRK_MODEL",
        "TRK_PROCESSOR",
        "VID_MODEL",
        "VID_PROCESSOR",
    ):
        monkeypatch.setattr(servers, name, None)
    monkeypatch.setattr(servers, "MODEL_META", {})


def test_default_startup_preloads_only_image_model(monkeypatch):
    """Default startup should not pay the memory cost for tracker/video."""

    class _FakeImageModel:
        def to(self, *_):
            return self

        def eval(self):
            return self

    image_model = _FakeImageModel()

    def fail_video_load(*_args, **_kwargs):
        raise AssertionError("video model should not load during default startup")

    _reset_model_globals(monkeypatch)
    monkeypatch.delenv("SAM3_PRELOAD", raising=False)
    monkeypatch.setenv("SAM3_SKIP_WEIGHTS_HASH", "1")
    monkeypatch.setattr(servers.Sam3Model, "from_pretrained", lambda *_a, **_kw: image_model)
    monkeypatch.setattr(servers.Sam3Processor, "from_pretrained", lambda *_a, **_kw: object())
    monkeypatch.setattr(servers.Sam3VideoModel, "from_pretrained", fail_video_load)

    servers._load_models()

    assert servers.SAM_MODEL is image_model
    assert servers.SAM_PROCESSOR is not None
    assert servers.TRK_MODEL is None
    assert servers.TRK_PROCESSOR is None
    assert servers.VID_MODEL is None
    assert servers.VID_PROCESSOR is None
    assert servers.MODEL_META["sam3_image_class"] == "_FakeImageModel"
    assert servers.MODEL_META["sam3_tracker_class"] is None
    assert servers.MODEL_META["sam3_video_class"] is None


def test_video_upgrade_releases_tracker_bound_to_old_image_model(monkeypatch):
    old_encoder = object()
    new_encoder = object()

    class _FakeImageModel:
        vision_encoder = old_encoder

    class _FakeTrackerModel:
        vision_encoder = old_encoder

    class _FakeDetectorModel:
        vision_encoder = new_encoder

    class _FakeVideoModel:
        detector_model = _FakeDetectorModel()

        def to(self, *_):
            return self

        def eval(self):
            return self

    _reset_model_globals(monkeypatch)
    monkeypatch.setenv("SAM3_SKIP_WEIGHTS_HASH", "1")
    monkeypatch.setattr(servers, "SAM_MODEL", _FakeImageModel())
    monkeypatch.setattr(servers, "SAM_PROCESSOR", object())
    monkeypatch.setattr(servers, "TRK_MODEL", _FakeTrackerModel())
    monkeypatch.setattr(servers, "TRK_PROCESSOR", object())
    monkeypatch.setattr(servers.Sam3VideoModel, "from_pretrained", lambda *_a, **_kw: _FakeVideoModel())
    monkeypatch.setattr(servers.Sam3VideoProcessor, "from_pretrained", lambda *_a, **_kw: object())

    servers._ensure_video_model()

    assert servers.SAM_MODEL is servers.VID_MODEL.detector_model
    assert servers.SAM_MODEL.vision_encoder is new_encoder
    assert servers.TRK_MODEL is None
    assert servers.TRK_PROCESSOR is None
    assert servers.MODEL_META["vision_encoder_shared"] is True
