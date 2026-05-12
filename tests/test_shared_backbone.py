"""真启动 servers 进程，验证 vision_encoder 在三处是同一 nn.Module。"""
import pytest
import os


@pytest.mark.slow_gpu
@pytest.mark.skipif(
    not os.path.isdir("/home/edward/research/SAM3-Gemma4-CUDA/models/facebook/sam3"),
    reason="SAM3 weights not on disk; skipping GPU integration test"
)
def test_vision_encoder_is_shared():
    import sys; sys.path.insert(0, "/home/edward/research/SAM3-Gemma4-CUDA")
    import servers
    servers._load_models()
    sam_ve = servers.SAM_MODEL.vision_encoder
    trk_ve = servers.TRK_MODEL.vision_encoder
    vid_ve = servers.VID_MODEL.detector_model.vision_encoder
    assert id(sam_ve) == id(trk_ve) == id(vid_ve), (
        f"vision_encoder not shared: SAM={id(sam_ve)} TRK={id(trk_ve)} VID={id(vid_ve)}"
    )
