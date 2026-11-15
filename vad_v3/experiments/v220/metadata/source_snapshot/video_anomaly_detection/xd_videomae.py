"""Local-window XD multi-label prior; no argmax or video-level Normal exit."""
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
try:
    from .xd_config import XD_CLASSES, XD_LABELS
    from . import vad_logger
except ImportError:
    from xd_config import XD_CLASSES, XD_LABELS
    import vad_logger

log = vad_logger.get('vad.xd.mae')


def time_windows(duration, window, stride):
    if duration <= 0:
        raise ValueError('Empty video')
    starts = list(np.arange(0, max(duration-window, 0)+1e-8, stride))
    tail = max(0., duration-window)
    if not starts or tail-starts[-1] > 1e-6:
        starts.append(tail)
    return [(float(s), float(min(s+window, duration))) for s in starts]


class XDVideoMAEPrior:
    def __init__(self, config, device):
        self.cfg, self.device = config, device
        self.model, report = VideoMAEForVideoClassification.from_pretrained(
            config.model_id, revision=config.model_revision, use_safetensors=True,
            output_loading_info=True)
        if any(report.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('XD VideoMAE did not load exactly: ' + str(report))
        if self.model.config.problem_type != 'multi_label_classification':
            raise ValueError('Expected a multi-label checkpoint')
        self.labels = [XD_LABELS[self.model.config.id2label[i]]
                       for i in range(self.model.config.num_labels)]
        if set(self.labels) != set((*XD_CLASSES, 'Normal')):
            raise ValueError('Unexpected XD checkpoint labels')
        self.model.to(device).eval()
        self.processor = VideoMAEImageProcessor.from_pretrained(
            config.processor_id, revision=config.processor_revision)
        self.loaded = True
        log.info('[XD-MAE] model=%s revision=%s processor_assumption=base_model:%s',
                 config.model_id, config.model_revision, config.processor_id)

    def predict(self, frames, timestamps, duration):
        """Uniform time sampling on the unfiltered frame pool; 16 frames/window.

        The base-model processor is an explicit fallback assumption. Per-window
        sigmoid outputs are saved, so later calibration need not rerun the GPU.
        """
        if not len(frames):
            raise ValueError('No decoded frames for XD prior')
        timestamps = np.asarray(timestamps)
        rows = []
        for start, end in time_windows(duration, self.cfg.window_seconds, self.cfg.stride_seconds):
            targets = np.linspace(start, max(start, end-1e-5), self.model.config.num_frames)
            right = np.clip(np.searchsorted(timestamps, targets), 0, len(timestamps)-1)
            left = np.maximum(right-1, 0)
            indices = np.where(abs(timestamps[left]-targets) < abs(timestamps[right]-targets), left, right)
            pixel_values = self.processor(list(frames[indices]), return_tensors='pt')['pixel_values'].to(self.device)
            with torch.inference_mode():
                scores = self.model(pixel_values=pixel_values).logits[0].sigmoid().float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite XD VideoMAE scores')
            rows.append(dict(t_start=start, t_end=end,
                             scores={c:float(scores[i]) for i,c in enumerate(self.labels)}))
        # Top local evidence avoids diluting brief events into a long-video mean.
        scores = {c:float(np.mean(sorted((w['scores'][c] for w in rows), reverse=True)[:3]))
                  for c in self.labels}
        log.info('[XD-PRIOR] windows=%d ranked=%s', len(rows),
                 sorted(scores.items(), key=lambda item:-item[1]))
        return dict(source=self.cfg.model_id, revision=self.cfg.model_revision,
                    processor_assumption='base_model_processor_not_verified_against_finetuning',
                    calibrated=False, scores=scores, windows=rows)
