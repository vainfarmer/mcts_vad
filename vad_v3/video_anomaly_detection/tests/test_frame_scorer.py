"""FrameScorer 测试 — guard clauses（无需真实 CLIP 模型）。"""
from unittest.mock import MagicMock

from config import VADConfig
from frame_scorer import FrameScorer


def _make_scorer(model=None):
    expert = MagicMock()
    expert.model = model  # None -> fallback paths
    cfg = VADConfig()
    return FrameScorer(expert, cfg, ["Arson", "Explosion", "Normal"])


class TestGetCategoryScores:
    def test_none_category_returns_none(self):
        fs = _make_scorer(model="loaded")
        vd = {"clip_features": "x"}
        assert fs.get_category_scores(vd, None) is None

    def test_normal_category_returns_none(self):
        fs = _make_scorer(model="loaded")
        vd = {"clip_features": "x"}
        assert fs.get_category_scores(vd, "Normal") is None

    def test_unknown_category_returns_none(self):
        fs = _make_scorer(model="loaded")
        vd = {"clip_features": "x"}
        assert fs.get_category_scores(vd, "Burglary") is None  # not in categories

    def test_no_clip_features_returns_none(self):
        fs = _make_scorer(model="loaded")
        vd = {}
        assert fs.get_category_scores(vd, "Arson") is None

    def test_no_model_returns_none(self):
        fs = _make_scorer(model=None)
        vd = {"clip_features": "x"}
        assert fs.get_category_scores(vd, "Arson") is None

    def test_caching(self):
        """第二次调用同一 category 应命中缓存（不再走模型路径）。"""
        fs = _make_scorer(model=None)  # model None -> returns None, but caches nothing
        vd = {}
        assert fs.get_category_scores(vd, "Arson") is None
        # 显式塞入缓存验证命中
        vd2 = {"_cat_scores_Arson": "cached_val"}
        assert fs.get_category_scores(vd2, "Arson") == "cached_val"


class TestComputeFrameScores:
    def test_no_frames_returns_none(self):
        fs = _make_scorer(model="loaded")
        import numpy as np
        vd = {"timestamps": np.array([]), "duration": 10, "clip_features": "x"}
        assert fs.compute_frame_scores(vd, [], "Arson") is None

    def test_no_clip_features_returns_none(self):
        fs = _make_scorer(model="loaded")
        import numpy as np
        vd = {"timestamps": np.array([0.0, 1.0]), "duration": 1, }  # no clip_features
        assert fs.compute_frame_scores(vd, [], "Arson") is None

    def test_no_model_returns_none(self):
        fs = _make_scorer(model=None)
        import numpy as np
        vd = {"timestamps": np.array([0.0, 1.0]), "duration": 1, "clip_features": "x"}
        assert fs.compute_frame_scores(vd, [], "Arson") is None


class TestClipScoreCurveCorrection:
    def test_no_scores_returns_unchanged(self):
        fs = _make_scorer(model="loaded")
        vd = {}  # no clip_features -> get_category_scores returns None
        anoms = [{"t_start": 0, "t_end": 5, "confidence": 0.9, "anomaly_type": "Arson"}]
        out = fs.clip_score_curve_correction(anoms, vd, "Arson")
        assert out is anoms  # unchanged (same list returned)

    def test_empty_anomalies_returns_empty(self):
        fs = _make_scorer(model="loaded")
        vd = {"_cat_scores_Arson": "fake"}  # provide cache so not None
        assert fs.clip_score_curve_correction([], vd, "Arson") == []
