"""models 测试 — DetectionResult/AnomalySegment 序列化与字段契约。"""
from models import AnomalySegment, DetectionResult, Segment, VideoData


class TestAnomalySegment:
    def test_to_dict_has_v33_keys(self):
        a = AnomalySegment(t_start=1.0, t_end=3.0, confidence=0.9, anomaly_type="Arson")
        d = a.to_dict()
        assert set(d.keys()) == {"t_start", "t_end", "confidence",
                                  "anomaly_type", "depth", "visits"}

    def test_from_dict_roundtrip(self):
        src = {"t_start": 1.0, "t_end": 3.0, "confidence": 0.9,
               "anomaly_type": "Arson", "depth": 2, "visits": 5}
        a = AnomalySegment.from_dict(src)
        assert a.t_start == 1.0 and a.confidence == 0.9
        assert a.depth == 2 and a.visits == 5
        assert a.to_dict() == src

    def test_from_dict_defaults_unknown_type(self):
        a = AnomalySegment.from_dict({"t_start": 0, "t_end": 1, "confidence": 0.5})
        assert a.anomaly_type == "unknown"


class TestDetectionResult:
    def test_to_dict_matches_batch_evaluate_schema(self):
        """to_dict 必须产出 batch_evaluate.py 写 JSON 的字段集（向后兼容）。"""
        r = DetectionResult(
            video_path="/v.mp4", duration=10.0, is_normal=False,
            anomalies=[AnomalySegment(1, 3, 0.9, "Arson")],
            frame_scores=[0.1, 0.2], timestamps=[1.0, 2.0],
        )
        d = r.to_dict()
        assert set(d.keys()) == {"video_path", "duration", "is_normal",
                                 "anomalies", "frame_scores", "timestamps"}
        assert d["anomalies"][0]["anomaly_type"] == "Arson"

    def test_frame_scores_defaults_none(self):
        r = DetectionResult(video_path="x", duration=1, is_normal=True)
        assert r.frame_scores is None
        assert r.timestamps is None
        assert r.anomalies == []

    def test_processing_time_not_in_to_dict(self):
        """processing_time 由调用方填，不在 to_dict 里（与 run.py 一致）。"""
        r = DetectionResult(video_path="x", duration=1, is_normal=True, processing_time=3.5)
        assert "processing_time" not in r.to_dict()


class TestVideoData:
    def test_dict_compatible(self):
        vd = VideoData()
        vd["clip_features"] = "fake"
        vd["duration"] = 10.0
        assert vd.get("clip_features") == "fake"
        assert "duration" in vd
