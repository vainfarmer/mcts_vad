"""VADConfig 测试 — 默认值对齐 v33、from_args/save/load。"""
import json
import os
import tempfile

from config import VADConfig, ScoringConfig, MCTSConfig


class TestDefaults:
    """默认值必须严格对齐当前 v33 行为（回归保护）。"""

    def test_scoring_defaults_match_v33(self):
        s = ScoringConfig()
        assert s.clip_base_weight == 0.35
        assert s.confidence_gate == 0.72
        assert s.gaussian_sigma_factor == 1.5
        assert s.gaussian_sigma_min == 1.0
        assert s.curve_smooth_seconds == 2.0
        assert s.curve_half_max_weight == 0.5
        assert s.curve_max_shift_ratio == 0.4
        assert s.curve_core_signal_floor == 0.3
        assert s.boundary_shrink_ratio == 0.35

    def test_mcts_defaults_match_v33(self):
        m = MCTSConfig()
        assert m.simulations == 12
        assert m.max_depth == 4
        assert m.padding_ratio == 0.75
        assert m.early_stop_agree == 4
        assert m.early_stop_min_sim == 5

    def test_pipeline_flat_accessors(self):
        c = VADConfig()
        assert c.fps == c.preprocess.fps
        assert c.segment_duration == c.preprocess.segment_duration
        assert c.mcts_simulations == c.mcts.simulations
        assert c.clip_model_name == c.clip.model_name
        assert c.max_frames == c.clip.max_frames


class TestFromArgs:
    def _args(self, **kw):
        class A: pass
        a = A()
        base = dict(fps=2, segment_duration=6.0, segment_overlap=0.25,
                    mcts_simulations=12, mcts_max_depth=4, device="cpu",
                    clip_model="ViT-L-14", clip_pretrained="openai",
                    max_frames=200, min_frames=3, fps_sample=3.0,
                    motion_threshold=1.0, min_motion_interval=2.0)
        base.update(kw)  # overrides take precedence
        for k, v in base.items():
            setattr(a, k, v)
        return a

    def test_from_args_maps_all(self):
        c = VADConfig.from_args(self._args(fps=4, segment_duration=8.0, device="mps"))
        assert c.preprocess.fps == 4
        assert c.preprocess.segment_duration == 8.0
        assert c.device == "mps"

    def test_from_args_missing_fields_use_defaults(self):
        class A: pass  # no attributes at all
        c = VADConfig.from_args(A())
        assert c.preprocess.fps == 2
        assert c.clip.model_name == "ViT-L-14"


class TestPersistence:
    def test_save_load_roundtrip(self):
        c = VADConfig()
        c.scoring.confidence_gate = 0.80
        c.mcts.simulations = 20
        c.device = "cuda"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            c.save(path)
            data = json.loads(open(path).read())
            assert data["scoring"]["confidence_gate"] == 0.80
            c2 = VADConfig.load(path)
            assert c2.scoring.confidence_gate == 0.80
            assert c2.mcts.simulations == 20
            assert c2.device == "cuda"
        finally:
            os.unlink(path)

    def test_load_partial_uses_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"scoring": {"confidence_gate": 0.66}}')
            path = f.name
        try:
            c = VADConfig.load(path)
            assert c.scoring.confidence_gate == 0.66
            assert c.scoring.clip_base_weight == 0.35  # default preserved
            assert c.mcts.simulations == 12  # default preserved
        finally:
            os.unlink(path)
