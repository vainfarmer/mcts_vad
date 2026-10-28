"""
CLIP 零样本异常分类专家 (v4)

v4 改进：
  - 升级模型: ViT-B-32 → ViT-L-14（更强的视觉表征）
  - 扩展 prompt: 每类别 2→6 条，覆盖更多视觉特征
  - 新增 get_category_distribution: 返回各类别概率分布，供 LLM 分类参考
  - 新增 get_top_anomaly_frames: CLIP 引导帧选择
  - 改进 normal prompt: 增加室内场景覆盖
"""

import numpy as np
from typing import Dict, Any, List

try:
    import torch
    import open_clip
    from PIL import Image
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    import vad_logger as _vl
    _log = _vl.get("vad.expert")
except Exception:
    import logging
    _log = logging.getLogger("vad.expert")


class VideoAnomalyExpert:

    def __init__(self, anomaly_categories: List[str],
                 clip_model_name: str = "ViT-L-14",
                 clip_pretrained: str = "openai",
                 device: str = "cpu"):
        pretrained = clip_pretrained
        self.categories = anomaly_categories
        self.device = device
        self.model = None
        self.preprocess = None
        self.text_features = None
        # v124 [COST]: drift-free per-video CLIP forward-pass counter (FM10).
        # Incremented on each encode_image call; pure instrumentation, never
        # read inside scoring, so it cannot alter any feature or detection.
        self.clip_call_count = 0

        if not HAS_CLIP:
            _log.warning("[AnomalyExpert] open_clip not available, running in fallback mode")
            return

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=pretrained, device=device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)
            self._precompute_text_features()
            _log.info(f"[AnomalyExpert] CLIP {clip_model_name} loaded on {device}")
        except Exception as e:
            self.model = None
            self.text_features = None
            _log.error(f"[AnomalyExpert] Failed to load CLIP: {e}, running in fallback mode")
            self.model = None

    CATEGORY_PROMPTS = {
        "Abuse": [
            "a surveillance video of physical abuse, a person hitting or hurting a defenseless victim who cannot fight back",
            "security camera footage of someone being physically abused, kicked, or beaten while on the ground",
            "CCTV footage of a stronger person repeatedly striking a helpless or restrained victim",
            "surveillance video showing a child or elderly person being physically mistreated",
            "security camera recording of domestic violence, one person attacking another who is cowering",
            "a surveillance scene where someone is being hurt by another person and unable to defend themselves",
        ],
        "Arrest": [
            "a surveillance video of police officers arresting and handcuffing a suspect on the ground",
            "security camera footage of law enforcement officers restraining and detaining a person",
            "CCTV footage showing uniformed police officers surrounding and subduing a suspect",
            "surveillance video of a police car with flashing lights and officers making an arrest",
            "security camera recording of multiple officers holding down a person and applying handcuffs",
            "a surveillance scene with police uniforms, patrol vehicles, and a person being taken into custody",
        ],
        "Arson": [
            "a surveillance video of arson, a person deliberately setting fire to a building with a lighter or match",
            "security camera footage of someone pouring flammable liquid and igniting a fire on purpose",
            "CCTV footage showing a person starting a fire that gradually spreads and grows from a single point",
            "surveillance video of deliberate fire-setting, with a person seen near the origin of flames",
            "security camera recording of someone lighting a fire at the base of a building or vehicle",
            "a surveillance scene where fire starts small at one spot and slowly grows, with a person fleeing the ignition point",
        ],
        "Assault": [
            "a surveillance video of a one-sided physical assault, an aggressive attacker beating a passive victim",
            "security camera footage of a violent attack where the victim is cowering, falling, or trying to escape",
            "CCTV footage of an assailant punching or kicking a person who is not fighting back",
            "surveillance video of a person being attacked from behind or knocked to the ground by an aggressor",
            "security camera recording of a one-sided violent encounter where only one person is attacking",
            "a surveillance scene showing an aggressive person striking someone who is defenseless or retreating",
        ],
        "Burglary": [
            "a surveillance video of burglary, a person breaking into a building by forcing open a door or window",
            "security camera footage of someone climbing through a window to enter a closed building at night",
            "CCTV footage of a person using tools to break a lock or pry open a door to gain entry",
            "surveillance video of forced entry into a building, with a person climbing walls or fences",
            "security camera recording of someone breaking glass to enter through a window of a building",
            "a surveillance scene showing a person sneaking into a closed building through an unauthorized entry point",
        ],
        "Explosion": [
            "a surveillance video where the scene INSTANTLY changes from calm to chaos with a bright flash and massive dust cloud in ONE frame",
            "security camera footage capturing the exact moment of a violent BLAST with debris and objects FLYING through the air simultaneously",
            "CCTV footage showing an INSTANTANEOUS detonation — one frame is normal, the next frame shows a huge fireball and shockwave",
            "surveillance video of a sudden explosion with a BLINDING FLASH, DUST and DEBRIS filling the entire frame at once",
            "security camera recording of a SPLIT-SECOND blast where the entire scene is engulfed in smoke and flying fragments",
            "a surveillance scene showing an IMMEDIATE catastrophic event with people thrown to the ground by an invisible shockwave force",
        ],
        "Fighting": [
            "a surveillance video of TWO people FACING each other and BOTH throwing punches — mutual physical combat",
            "security camera footage of a BRAWL where ALL participants are actively HITTING each other, no one is passive",
            "CCTV footage of two people GRAPPLING and WRESTLING, BOTH pulling and pushing each other aggressively",
            "surveillance video of MUTUAL combat — both people are in aggressive fighting stances and EXCHANGING BLOWS",
            "security camera recording of a FISTFIGHT where you can see BOTH sides throwing punches at each other alternately",
            "a surveillance scene showing TWO ACTIVE combatants — neither is running away, both are engaged in physical combat",
        ],
        "RoadAccidents": [
            "a surveillance video of a road accident, two vehicles crashing into each other on a street",
            "security camera footage of a car collision at an intersection with vehicle damage and debris",
            "CCTV footage of a traffic accident where a vehicle hits a pedestrian or another car",
            "surveillance video of a vehicle losing control, flipping over, or skidding off the road",
            "security camera recording of a high-speed car crash with vehicles spinning and parts flying",
            "a surveillance scene at a road showing vehicles colliding, with shattered glass and crumpled metal",
        ],
        "Robbery": [
            "a surveillance video of armed robbery, a person pointing a gun or knife at a victim demanding valuables",
            "security camera footage of a holdup where an armed robber threatens a person face-to-face",
            "CCTV footage of robbery with a weapon visible, the victim handing over money or belongings under threat",
            "surveillance video of a robber wielding a weapon and forcibly taking items from a terrified victim",
            "security camera recording of an armed person threatening a store clerk or pedestrian for money",
            "a surveillance scene showing a face-to-face confrontation where a weapon-wielding person demands valuables",
        ],
        "Shooting": [
            "a surveillance video of a shooting, a person firing a gun with muzzle flash visible",
            "security camera footage of gunfire with people suddenly falling to the ground or fleeing in panic",
            "CCTV footage of a shooting incident where a crowd scatters and runs in all directions from gunshots",
            "surveillance video showing a person with a firearm shooting at people, victims dropping to the ground",
            "security camera recording of a mass panic with people running and ducking, suggesting gunfire",
            "a surveillance scene where people suddenly collapse or scatter in terror, indicating a shooting event",
        ],
        "Shoplifting": [
            "a surveillance video inside a retail store of a person concealing merchandise under their clothing",
            "security camera footage in a shop of someone hiding a product in their bag while looking around nervously",
            "CCTV footage inside a store showing a person stuffing items into a bag or pocket and walking toward the exit",
            "surveillance video of shoplifting in a retail store with shelves of products, a person secretly taking items",
            "security camera recording inside a shop of a person picking up merchandise and hiding it in their jacket",
            "a store surveillance scene where a customer is sneaking products off shelves into a bag without paying",
        ],
        "Stealing": [
            "a surveillance video of a person REACHING OUT and GRABBING someone's bag or phone, then RUNNING AWAY quickly",
            "security camera footage of someone PICKING UP an unattended purse or laptop from a table and WALKING AWAY with it",
            "CCTV footage of a person SNATCHING an item from someone's hand and FLEEING on foot",
            "surveillance video of a PICKPOCKET sliding their hand into someone's pocket to TAKE a wallet without being noticed",
            "security camera recording of someone OPENING a parked car door and TAKING items from inside, then leaving quickly",
            "a surveillance scene where a person GRABS unattended belongings and RUNS — the key action is TAKING and FLEEING with an object",
        ],
        "Vandalism": [
            "a surveillance video of vandalism, a person smashing a car window or storefront with a bat or rock",
            "security camera footage of someone spray-painting graffiti on walls or property",
            "CCTV footage of a person kicking, smashing, or deliberately destroying street furniture or public property",
            "surveillance video of deliberate property destruction, someone throwing rocks at windows or vehicles",
            "security camera recording of a person vandalizing a parked car by scratching, denting, or breaking it",
            "a surveillance scene showing someone destroying property in place, breaking objects without entering a building",
        ],
    }

    NORMAL_PROMPTS = [
        "normal daily activity in a surveillance camera, people walking and going about their routine",
        "a calm and uneventful surveillance camera scene with no criminal activity",
        "security camera footage of ordinary pedestrians and normal traffic, nothing unusual happening",
        "surveillance video of people shopping normally inside a store, browsing shelves peacefully",
        "security camera footage of an empty hallway, parking lot, or building entrance with no activity",
        "a normal indoor surveillance scene showing people sitting, working, or having conversations",
    ]

    def _precompute_text_features(self):
        # v93: learned prompt-ensemble weights (Loop 12 research, CoOp-lite).
        # Trained via MIL top-k video-level loss on the UCF-Crime train split's
        # video-level labels only (no frame-level GT used or needed). Replaces
        # simple mean-ensembling of the 6 hand-crafted prompts/category with a
        # learned weighted combination. Validated on the HELD-OUT test set
        # (not just training data) before integration: 12/13 categories show
        # improved video-level separation, avg 0.6230->0.6464 (+3.8%). This is
        # the tractable simplification of full CoOp (which would require
        # injecting learnable continuous embeddings into CLIP's token layer -
        # high complexity/risk) - operates STRICTLY upstream of the fusion/
        # boundary detection logic (only changes which text embedding IS the
        # category's representation, computed once and cached exactly like
        # the original mean-ensemble). Falls back to uniform mean if the
        # weights file is missing.
        import json as _json
        from pathlib import Path as _PP
        _weights_path = _PP(__file__).parent / 'prompt_weights.json'
        _learned_weights = {}
        if _weights_path.exists():
            try:
                with open(_weights_path) as _wf:
                    _learned_weights = _json.load(_wf)
                _log.info(f"[AnomalyExpert] Loaded learned prompt weights from {_weights_path}")
            except Exception as _we:
                _log.warning(f"[AnomalyExpert] Failed to load prompt weights: {_we}")

        category_features = []
        for cat in self.categories:
            prompts = self.CATEGORY_PROMPTS.get(cat, [
                f"a surveillance camera video of {cat.lower()}"
            ])
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                _w = _learned_weights.get(cat)
                if _w is not None and len(_w) == feats.shape[0]:
                    _wt = torch.tensor(_w, device=feats.device, dtype=feats.dtype).unsqueeze(-1)
                    ensemble = (feats * _wt).sum(dim=0)
                else:
                    ensemble = feats.mean(dim=0)
                ensemble = ensemble / ensemble.norm()
            category_features.append(ensemble)

        normal_tokens = self.tokenizer(self.NORMAL_PROMPTS).to(self.device)
        with torch.no_grad():
            normal_feats = self.model.encode_text(normal_tokens)
            normal_feats = normal_feats / normal_feats.norm(dim=-1, keepdim=True)
            normal_ensemble = normal_feats.mean(dim=0)
            normal_ensemble = normal_ensemble / normal_ensemble.norm()
        category_features.append(normal_ensemble)

        self.text_features = torch.stack(category_features, dim=0)



    def precompute_video_features(self, all_frames: np.ndarray, batch_size: int = 32):
        if self.model is None or len(all_frames) == 0:
            return None

        use_fp16 = self.device == 'cuda'
        all_features = []
        for i in range(0, len(all_frames), batch_size):
            batch = all_frames[i:i + batch_size]
            images = [self.preprocess(Image.fromarray(f)) for f in batch]
            image_tensor = torch.stack(images).to(self.device)
            self.clip_call_count += 1  # v124 [COST]
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=use_fp16):
                features = self.model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def _encode_frames(self, frames: np.ndarray, max_frames: int = 8):
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]

        images = []
        for frame in frames:
            img = Image.fromarray(frame)
            images.append(self.preprocess(img))

        image_tensor = torch.stack(images).to(self.device)
        self.clip_call_count += 1  # v124 [COST]
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_category_distribution(self, frame_indices, precomputed_features, top_k=5):
        """返回给定帧集合的类别概率分布，供 LLM 分类参考"""
        if self.model is None or precomputed_features is None or not frame_indices:
            return {}

        with torch.no_grad():
            features = precomputed_features[frame_indices]
            logit_scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = features @ self.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1)
            mean_probs = probs.mean(dim=0).cpu().numpy()

        cat_probs = {}
        for i, cat in enumerate(self.categories):
            cat_probs[cat] = float(mean_probs[i])

        sorted_cats = sorted(cat_probs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_cats[:top_k])

    def get_top_anomaly_frames(self, frame_indices, precomputed_features, n=16):
        """返回异常分数最高的 n 帧的索引，用于 Stage 6 CLIP 引导帧选择"""
        if self.model is None or precomputed_features is None or not frame_indices:
            return frame_indices[:n] if frame_indices else []

        with torch.no_grad():
            features = precomputed_features[frame_indices]
            logit_scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = features @ self.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1)
            top3_anomaly = probs[:, :-1].topk(min(3, probs.shape[1]-1), dim=-1).values.sum(dim=-1)
            normal_prob = probs[:, -1]
            anomaly_scores = (top3_anomaly / (top3_anomaly + normal_prob)).cpu().numpy()

        ranked = np.argsort(anomaly_scores)[::-1][:n]
        ranked_sorted = sorted(ranked)
        return [frame_indices[i] for i in ranked_sorted]

    def batch_score_segments(self, segments, precomputed_features,
                             max_frames_per_seg=8):
        if self.model is None or precomputed_features is None:
            return [{'anomaly_score': 0.0, 'predicted_type': 'unknown'}
                    for _ in segments]

        all_indices = []
        segment_slices = []
        for seg in segments:
            idx = seg['frame_indices']
            if len(idx) > max_frames_per_seg:
                sample = np.linspace(0, len(idx) - 1, max_frames_per_seg, dtype=int)
                idx = [seg['frame_indices'][i] for i in sample]
            start = len(all_indices)
            all_indices.extend(idx)
            segment_slices.append((start, len(all_indices)))

        if not all_indices:
            return [{'anomaly_score': 0.0, 'predicted_type': 'unknown'}
                    for _ in segments]

        with torch.no_grad():
            features = precomputed_features[all_indices]
            logit_scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = features @ self.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1)
            top3_anomaly = probs[:, :-1].topk(min(3, probs.shape[1]-1), dim=-1).values.sum(dim=-1)
            normal_prob = probs[:, -1]
            anomaly_scores = top3_anomaly / (top3_anomaly + normal_prob)

        results = []
        for start, end in segment_slices:
            seg_anomaly = anomaly_scores[start:end]
            seg_probs = probs[start:end]
            anomaly_score = float(seg_anomaly.mean())
            mean_p = seg_probs.mean(dim=0).cpu().numpy()
            best_cat_idx = int(mean_p[:-1].argmax())
            results.append({
                'anomaly_score': anomaly_score,
                'predicted_type': self.categories[best_cat_idx],
            })
        return results

    async def score_segment(self, frames: np.ndarray,
                            t_start: float, t_end: float,
                            frame_indices=None,
                            precomputed_features=None) -> Dict[str, Any]:
        if self.model is None or len(frames) == 0:
            return self._fallback_result(t_start, t_end)

        if precomputed_features is not None and frame_indices is not None:
            idx = frame_indices
            if len(idx) > 8:
                sample_idx = np.linspace(0, len(idx) - 1, 8, dtype=int)
                idx = [frame_indices[i] for i in sample_idx]
            image_features = precomputed_features[idx]
        else:
            image_features = self._encode_frames(frames)

        similarity = image_features @ self.text_features.T
        logit_scale = self.model.logit_scale.exp().detach().clamp(max=100.0)
        probs = torch.softmax(similarity * logit_scale, dim=-1)

        top3_anomaly = probs[:, :-1].topk(min(3, probs.shape[1]-1), dim=-1).values.sum(dim=-1)
        normal_prob = probs[:, -1]
        frame_anomaly_scores = (top3_anomaly / (top3_anomaly + normal_prob)).cpu().numpy()

        anomaly_score = float(frame_anomaly_scores.mean())

        mean_probs = probs.mean(dim=0).cpu().numpy()
        category_probs = mean_probs[:-1]
        best_cat_idx = int(category_probs.argmax())
        predicted_type = self.categories[best_cat_idx]

        category_scores = {
            cat: float(category_probs[i])
            for i, cat in enumerate(self.categories)
        }

        peak_frame_idx = int(frame_anomaly_scores.argmax())
        n_frames = max(len(frames) - 1, 1)
        peak_time = t_start + peak_frame_idx / n_frames * (t_end - t_start)

        return {
            'anomaly_score': anomaly_score,
            'predicted_type': predicted_type,
            'category_scores': category_scores,
            'peak_frame_idx': peak_frame_idx,
            'peak_time': float(peak_time),
        }

    def _fallback_result(self, t_start, t_end):
        return {
            'anomaly_score': 0.0,
            'predicted_type': 'unknown',
            'category_scores': {},
            'peak_frame_idx': 0,
            'peak_time': t_start,
        }
