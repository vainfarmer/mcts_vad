"""XD-specific settings; the UCF checkpoint and tuned duration rules stay separate."""
from dataclasses import dataclass

XD_CLASSES = ('Fighting', 'Shooting', 'Riot', 'Abuse', 'CarAccident', 'Explosion')
XD_LABELS = {'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
             'B5': 'Abuse', 'B6': 'CarAccident', 'G': 'Explosion'}


@dataclass
class XDConfig:
    model_id: str = 'KingTechnician/videomae-small-finetuned-kinetics-finetuned-xd-violence-multilabel'
    model_revision: str = '54875f0b60e8acad7f1f8934e3f20ca6118f5b36'
    # The target repository has no processor. This is an explicit experimental
    # assumption, not a claim that the fine-tuning author's preprocessing is known.
    processor_id: str = 'MCG-NJU/videomae-small-finetuned-kinetics'
    processor_revision: str = '1bf2b82c809609f5df1ea7648aee577e2e486b63'
    window_seconds: float = 8.0
    stride_seconds: float = 4.0
    windows_per_class: int = 2
    coverage_windows: int = 2
    frames_per_request: int = 16
    nms_iou: float = 0.5
    boundary_max_shift_ratio: float = 0.2
    boundary_min_contrast: float = 0.05
    max_frame_side: int = 448

    def __post_init__(self):
        if not 0 < self.stride_seconds <= self.window_seconds:
            raise ValueError('XD stride must be positive and no larger than window')
        if self.windows_per_class < 1 or self.coverage_windows < 0:
            raise ValueError('Invalid XD candidate budget')
        if self.frames_per_request < 2 or not 0 < self.nms_iou <= 1:
            raise ValueError('Invalid XD evidence/NMS settings')
        if not 0 <= self.boundary_max_shift_ratio < 0.5 or self.max_frame_side < 224:
            raise ValueError('Invalid XD boundary/image settings')
