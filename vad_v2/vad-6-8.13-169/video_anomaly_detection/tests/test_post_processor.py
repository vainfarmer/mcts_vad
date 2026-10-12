"""post_processor 纯函数测试 — 合并/统一/收窄逻辑。"""
import numpy as np
from post_processor import global_merge, unify_anomaly_types, narrow_short_events


def _seg(s, e, conf=0.9, t="Arson"):
    return {"t_start": s, "t_end": e, "confidence": conf, "anomaly_type": t}


class TestGlobalMerge:
    def test_empty_returns_empty(self):
        assert global_merge([]) == []

    def test_merge_overlapping_same_type(self):
        merged = global_merge([_seg(0, 5), _seg(3, 8)])
        assert len(merged) == 1
        assert merged[0]["t_start"] == 0 and merged[0]["t_end"] == 8
        assert merged[0]["confidence"] == 0.9  # max

    def test_merge_adjacent_within_gap(self):
        # gap = 2 <= same_type_gap(2.0) -> merge
        merged = global_merge([_seg(0, 4), _seg(6, 10)])
        assert len(merged) == 1
        assert merged[0]["t_start"] == 0 and merged[0]["t_end"] == 10

    def test_no_merge_far_apart_same_type(self):
        merged = global_merge([_seg(0, 2), _seg(20, 24)])
        assert len(merged) == 2

    def test_merge_diff_type_high_iou(self):
        # IoU high (>0.3) -> merge despite different type
        merged = global_merge([_seg(0, 10, t="Arson"), _seg(5, 15, t="Explosion")])
        assert len(merged) == 1

    def test_no_merge_diff_type_low_iou(self):
        # barely overlap, union large -> IoU low -> keep separate
        merged = global_merge([_seg(0, 5, t="Arson"), _seg(4, 20, t="Explosion")])
        assert len(merged) == 2

    def test_preserves_higher_confidence_trace(self):
        merged = global_merge([_seg(0, 5, conf=0.9), _seg(3, 8, conf=0.7)])
        # depth/visits come from higher-confidence segment
        assert merged[0]["confidence"] == 0.9

    def test_iterative_merge(self):
        # a-b overlap, b-c overlap -> all three merge into one
        merged = global_merge([_seg(0, 4), _seg(3, 7), _seg(6, 10)])
        assert len(merged) == 1
        assert merged[0]["t_start"] == 0 and merged[0]["t_end"] == 10


class TestUnifyAnomalyTypes:
    def test_single_unchanged(self):
        a = [_seg(0, 5, t="Arson")]
        assert unify_anomaly_types(a) == a

    def test_same_type_unchanged(self):
        a = [_seg(0, 5, t="Arson"), _seg(6, 9, t="Arson")]
        out = unify_anomaly_types(a)
        assert all(x["anomaly_type"] == "Arson" for x in out)

    def test_winner_by_confidence_times_duration(self):
        # Arson: 0.9*5 = 4.5 ; Stealing: 0.5*3 = 1.5 -> Arson wins
        a = [_seg(0, 5, conf=0.9, t="Arson"), _seg(6, 9, conf=0.5, t="Stealing")]
        out = unify_anomaly_types(a)
        assert all(x["anomaly_type"] == "Arson" for x in out)

    def test_does_not_mutate_input(self):
        a = [_seg(0, 5, t="Arson"), _seg(6, 9, t="Stealing")]
        orig_types = [x["anomaly_type"] for x in a]
        unify_anomaly_types(a)
        assert [x["anomaly_type"] for x in a] == orig_types


class TestNarrowShortEvents:
    def _vd(self, motion, n=20, dur=20.0):
        return {
            "timestamps": np.linspace(0, dur, n),
            "duration": dur,
            "motion_energy": np.asarray(motion, dtype=float),
        }

    def test_no_motion_returns_unchanged(self):
        a = [_seg(0, 12)]
        assert narrow_short_events(a, {"timestamps": np.linspace(0, 20, 20),
                                       "duration": 20.0}) == a

    def test_short_segment_unchanged(self):
        a = [_seg(0, 8)]  # <= 10s -> unchanged
        out = narrow_short_events(a, self._vd(np.zeros(20)))
        assert out[0]["t_start"] == 0 and out[0]["t_end"] == 8

    def test_long_segment_with_peak_narrows(self):
        # long segment, sharp motion peak at t~10 -> narrowed around peak
        motion = np.full(20, 0.1)
        motion[10] = 5.0  # strong peak
        a = [_seg(0, 20)]
        out = narrow_short_events(a, self._vd(motion))
        narrowed = out[0]
        # narrowed window should be ~ peak(10) ± 5 = [5, 15], shorter than original
        assert (narrowed["t_end"] - narrowed["t_start"]) < 20
        assert narrowed["t_start"] >= 5.0 - 0.01
