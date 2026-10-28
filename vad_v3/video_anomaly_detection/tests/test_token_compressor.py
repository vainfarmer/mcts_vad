"""SemanticFrameSampler.compute_target_frames 纯数学测试（无需 CLIP）。"""
from token_compressor import SemanticFrameSampler


def _sampler():
    return SemanticFrameSampler(anomaly_expert=None, tau=0.92,
                               max_frames=200, min_frames=3, fps_sample=3.0)


class TestComputeTargetFrames:
    def test_scales_with_duration(self):
        s = _sampler()
        short = s.compute_target_frames(duration=2.0)
        long = s.compute_target_frames(duration=20.0)
        assert long > short

    def test_respects_min_frames(self):
        s = _sampler()
        # very short duration -> still at least min_frames
        assert s.compute_target_frames(duration=0.1) >= 3

    def test_respects_max_for_depth(self):
        s = _sampler()
        # shallow depth: max = max_frames / (1 + 0*0.3) = 200
        n = s.compute_target_frames(duration=1000.0, depth=0)
        assert n <= 200

    def test_deeper_depth_lower_cap(self):
        s = _sampler()
        shallow = s.compute_target_frames(duration=1000.0, depth=0)
        deep = s.compute_target_frames(duration=1000.0, depth=3)
        assert deep <= shallow

    def test_n_shrinks_increases_density(self):
        s = _sampler()
        base = s.compute_target_frames(duration=5.0, n_shrinks=0)
        dense = s.compute_target_frames(duration=5.0, n_shrinks=2)
        assert dense >= base
