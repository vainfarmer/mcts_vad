"""
VideoMAE UCF-Crime 分类器 (v2 — with Test-Time Augmentation)

使用在 UCF-Crime 上微调的 VideoMAE-large 进行 14 类分类。
v2: 多次前向传播 + 平均 logits 提高分类稳定性
"""

import numpy as np
import torch
from PIL import Image
from typing import Dict

try:
    import vad_logger as _vl
    _log = _vl.get("vad.videomae")
except Exception:
    import logging
    _log = logging.getLogger("vad.videomae")


class VideoMAEClassifier:

    ID2LABEL = {
        0: 'Abuse', 1: 'Arrest', 2: 'Arson', 3: 'Assault', 4: 'Burglary',
        5: 'Explosion', 6: 'Fighting', 7: 'Normal', 8: 'RoadAccidents',
        9: 'Robbery', 10: 'Shooting', 11: 'Shoplifting', 12: 'Stealing',
        13: 'Vandalism',
    }

    def __init__(self, model_name: str = "OPear/videomae-large-finetuned-UCF-Crime",
                 device: str = "cpu", num_frames: int = 16):
        self.device = device
        self.num_frames = num_frames
        self.model = None

        try:
            self._load_model(model_name)
            _log.info(f"[VideoMAE] Loaded on {device}")
        except Exception as e:
            self.model = None
            _log.error(f"[VideoMAE] Failed to load: {e}")
            raise

    def _load_model(self, model_name):
        from transformers import VideoMAEForVideoClassification, VideoMAEConfig
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(model_name, 'config.json')
        config_dict = json.load(open(config_path))
        config_dict['num_labels'] = 14
        config_dict['id2label'] = {int(k): v for k, v in config_dict['id2label'].items()}
        config_dict['label2id'] = {v: int(k) for k, v in config_dict['id2label'].items()}

        config = VideoMAEConfig(**config_dict)
        self.model = VideoMAEForVideoClassification(config)

        model_path = hf_hub_download(model_name, 'model.safetensors')
        state_dict = load_file(model_path)

        expected = set(self.model.state_dict())
        new_state_dict = {}
        for k, v in state_dict.items():
            mapped = k
            for old, new in (('.q_bias', '.query.bias'), ('.v_bias', '.value.bias')):
                if k.endswith(old) and k not in expected:
                    mapped = k[:-len(old)] + new
            new_state_dict[mapped] = v
        for key in expected:
            if key.endswith('.key.bias') and key not in new_state_dict:
                qkey = key.replace('.key.bias', '.query.bias')
                if qkey in new_state_dict:
                    new_state_dict[key] = torch.zeros_like(new_state_dict[qkey])
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        n = len(frames)
        if n >= self.num_frames:
            idx = np.linspace(0, n - 1, self.num_frames, dtype=int)
        else:
            idx = np.array(list(range(n)) + [n - 1] * (self.num_frames - n))
        sampled = frames[idx]

        resized = []
        for f in sampled:
            img = Image.fromarray(f).resize((224, 224), Image.BILINEAR)
            resized.append(np.array(img))

        tensor = torch.from_numpy(np.stack(resized)).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _preprocess_with_offset(self, frames: np.ndarray, offset: int) -> torch.Tensor:
        """Sample 16 frames with a temporal offset for TTA."""
        n = len(frames)
        if n < self.num_frames:
            return self._preprocess(frames)

        step = n / self.num_frames
        idx = np.array([int(min(offset + i * step, n - 1)) for i in range(self.num_frames)])
        idx = np.clip(idx, 0, n - 1)
        sampled = frames[idx]

        resized = []
        for f in sampled:
            img = Image.fromarray(f).resize((224, 224), Image.BILINEAR)
            resized.append(np.array(img))

        tensor = torch.from_numpy(np.stack(resized)).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def classify(self, frames: np.ndarray, top_k: int = 3, tta: bool = True) -> Dict:
        if self.model is None or len(frames) == 0:
            return {'category': 'unknown', 'confidence': 0.0, 'top_k': []}

        with torch.no_grad():
            if tta and len(frames) >= self.num_frames * 2:
                logits = self._classify_tta(frames)
            else:
                tensor = self._preprocess(frames)
                logits = self.model(tensor).logits[0]

            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        ranked = np.argsort(probs)[::-1]
        top_results = [(self.ID2LABEL[i], float(probs[i])) for i in ranked[:top_k]]

        best_id = ranked[0]
        best_label = self.ID2LABEL[best_id]

        return {
            'category': best_label,
            'confidence': float(probs[best_id]),
            'top_k': top_results,
            'all_probs': {self.ID2LABEL[i]: float(probs[i]) for i in range(14)},
        }


    def sliding_window_scores(self, frames: np.ndarray,
                              window_size: int = 32, stride: int = 16):
        """滑动窗口扫描：返回 (centers, anomaly_scores)。"""
        if self.model is None or len(frames) == 0:
            return [], []
        n = len(frames)
        if n < self.num_frames:
            result = self.classify(frames, tta=False)
            normal_prob = result['all_probs'].get('Normal', 0.0)
            return [n // 2], [1.0 - normal_prob]
        windows = []
        for start in range(0, max(1, n - window_size + 1), stride):
            end = min(start + window_size, n)
            if end - start < self.num_frames:
                break
            windows.append((start, end))
        if not windows:
            result = self.classify(frames, tta=False)
            normal_prob = result['all_probs'].get('Normal', 0.0)
            return [n // 2], [1.0 - normal_prob]
        centers = []
        scores = []
        tensors = [self._preprocess(frames[s:e]) for s, e in windows]
        with torch.no_grad():
            for i, tensor in enumerate(tensors):
                logits = self.model(tensor).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                normal_prob = float(probs[7])
                s, e = windows[i]
                centers.append((s + e) // 2)
                scores.append(1.0 - normal_prob)
        return centers, scores

    def _classify_tta(self, frames: np.ndarray) -> torch.Tensor:
        """Test-time augmentation: multiple temporal crops, average logits."""
        n = len(frames)
        all_logits = []

        # Crop 1: uniform sampling (standard)
        tensor = self._preprocess(frames)
        all_logits.append(self.model(tensor).logits[0])

        # Crop 2: offset by half-step
        half_step = max(1, n // (self.num_frames * 2))
        tensor = self._preprocess_with_offset(frames, half_step)
        all_logits.append(self.model(tensor).logits[0])

        # Crop 3: first 60% of frames
        end_60 = max(self.num_frames, int(n * 0.6))
        tensor = self._preprocess(frames[:end_60])
        all_logits.append(self.model(tensor).logits[0])

        # Crop 4: last 60% of frames
        start_40 = max(0, n - end_60)
        tensor = self._preprocess(frames[start_40:])
        all_logits.append(self.model(tensor).logits[0])

        # Crop 5: middle 60%
        mid_start = max(0, int(n * 0.2))
        mid_end = min(n, int(n * 0.8))
        if mid_end - mid_start >= self.num_frames:
            tensor = self._preprocess(frames[mid_start:mid_end])
            all_logits.append(self.model(tensor).logits[0])

        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits
