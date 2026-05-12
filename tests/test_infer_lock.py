"""测试 _INFER_LOCK 是 threading.Lock，三个 endpoint 函数在调用 model 之前会进入锁。"""
import threading
import servers

def test_infer_lock_is_threading_lock():
    assert isinstance(servers._INFER_LOCK, type(threading.Lock()))

def test_endpoints_acquire_lock(monkeypatch):
    """concurrent calls into detect() must serialize on _INFER_LOCK."""
    import time
    # Mock heavy model + processor so the test does not need GPU/weights.
    class _FakeProc:
        def __call__(self, **kw): return _FakeBatched()
        def post_process_instance_segmentation(self, *a, **kw):
            return [{"masks": None, "scores": None}]
    class _FakeBatched(dict):
        def to(self, *_): return self
        def get(self, k, d=None): return d
    class _FakeModel:
        def __call__(self, **kw):
            time.sleep(0.2)
            return object()
    monkeypatch.setattr(servers, "SAM_MODEL", _FakeModel())
    monkeypatch.setattr(servers, "SAM_PROCESSOR", _FakeProc())
    # Bypass image decode
    monkeypatch.setattr(servers, "_decode_b64_image", lambda b: _StubImg())
    class _StubImg:
        size = (8, 8)
    from servers import DetectRequest, detect

    holding = []
    barrier = threading.Barrier(2)
    def worker():
        barrier.wait()
        t0 = time.monotonic()
        try:
            detect(DetectRequest(image_b64="", prompt="x"))
        except Exception:
            pass
        holding.append(time.monotonic() - t0)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    # If lock works, total wall ≥ 2 × per-call sleep. The second caller waits.
    assert max(holding) >= 0.35, f"expected serialization, got {holding}"
