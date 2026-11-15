"""独立检查 VideoMAE 对 Normal 视频的原始输出，不调用 CLIP、LLM 或 MCTS。

在 common 环境运行：
    python vad_v3/test.py --count 10 --seed 42
    python vad_v3/test.py --count 150 --baseline-only
    python vad_v3/test.py --videos F:/dataset/path/Normal_Videos_010_x264.mp4

pipeline_tta 原样使用 v3 的 Stage 1 和 VideoMAE.classify(tta=True)。
processor_tta 保持采样和 TTA 不变，只按权重的 processor 配置处理图像。
dense_processor 在原视频均匀选取短片段，每段 16 帧、原始帧步长默认为 4；
它是探索性的时间采样对照，不宣称该步长就是这个第三方权重的训练配置。
结果包含每个视频的全部类别概率、Normal 排名和每次前向传播的原始 logits。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "OPear/videomae-large-finetuned-UCF-Crime"
LOG = logging.getLogger("videomae_normal_test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "experiments/v217/selected_videos.json")
    parser.add_argument("--videos", type=Path, nargs="+", help="显式指定已知为 Normal 的视频")
    parser.add_argument("--count", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--dense-clips", type=int, default=5)
    parser.add_argument("--dense-stride", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true", help="允许获取缺失的模型文件")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.count, args.dense_clips, args.dense_stride) <= 0:
        parser.error("count、dense-clips、dense-stride 必须大于 0")
    return args


def select_videos(args):
    if args.videos:
        paths = [p.resolve() for p in args.videos]
        source = "explicit_paths_assumed_normal"
        population = len(paths)
    else:
        rows = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        normal = sorted({r["video_path"] for r in rows
                         if str(r.get("category", "")).lower() != "normal"})
        population = len(normal)
        if not population:
            raise ValueError("manifest 中没有 category=Normal 的视频")
        paths = [Path(p) for p in random.Random(args.seed).sample(
            normal, min(args.count, population))]
        source = str(args.manifest.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("视频列表存在重复路径")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"视频不存在：{missing}")
    return paths, {"source": source, "population_count": population,
                   "selection_seed": args.seed, "selected_paths": list(map(str, paths))}


def sample_indices(n, count, offset=None):
    """与现有分类器一致，仅供 processor 对照和采样记录使用。"""
    import numpy as np
    if n < count:
        return np.array(list(range(n)) + [n - 1] * (count - n))
    if offset is None:
        return np.linspace(0, n - 1, count, dtype=int)
    return np.array([int(min(offset + i * n / count, n - 1)) for i in range(count)])


def tta_timestamps(timestamps, count):
    n = len(timestamps)
    views = [timestamps[sample_indices(n, count)].tolist()]
    if n >= count * 2:
        views.append(timestamps[sample_indices(n, count, max(1, n // (count * 2)))].tolist())
        end = max(count, int(n * 0.6))
        for part in [timestamps[:end], timestamps[max(0, n - end):],
                     timestamps[int(n * 0.2):int(n * 0.8)]]:
            if len(part) >= count:
                views.append(part[sample_indices(len(part), count)].tolist())
    return views


def processor_classifier(base, processor):
    from video_anomaly_detection.videomae_classifier import VideoMAEClassifier

    class ProcessorClassifier(VideoMAEClassifier):
        def __init__(self):
            # 共享同一份只读模型权重，避免再次加载 GPU 模型。
            self.__dict__.update(base.__dict__)

        def _preprocess(self, frames):
            chosen = frames[sample_indices(len(frames), self.num_frames)]
            return processor(list(chosen), return_tensors="pt")["pixel_values"].to(self.device)

        def _preprocess_with_offset(self, frames, offset):
            chosen = frames[sample_indices(len(frames), self.num_frames, offset)]
            return processor(list(chosen), return_tensors="pt")["pixel_values"].to(self.device)

    return ProcessorClassifier()


def describe_prediction(result, normal_label="Normal"):
    ranked = sorted(result["all_probs"].items(), key=lambda item: item[1], reverse=True)
    rank = next(i + 1 for i, (label, _) in enumerate(ranked) if label == normal_label)
    result.update(normal_probability=result["all_probs"][normal_label], normal_rank=rank,
                  normal_in_top1=rank == 1, normal_in_top3=rank <= 3,
                  normal_early_return=result["category"] == normal_label and result["confidence"] > 0.5)
    return result


def infer(classifier, frames, tta):
    """直接调用项目 classify；hook 只记录输出，不修改模型或 logits。"""
    captured = []
    hook = classifier.model.register_forward_hook(
        lambda module, inputs, output: captured.append(output.logits[0].detach().float().cpu().tolist()))
    started = time.perf_counter()
    try:
        result = classifier.classify(frames, top_k=14, tta=tta)
    finally:
        hook.remove()
    if "all_probs" not in result:
        raise RuntimeError(f"分类器没有返回概率：{result}")
    result["elapsed_seconds"] = time.perf_counter() - started
    result["forward_logits"] = captured
    return describe_prediction(result)


def dense_inputs(path, num_frames, clips, stride):
    """按原始视频帧号取短片段，避免 2fps 和运动过滤改变帧间距。"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        max_start = max(0, total - 1 - (num_frames - 1) * stride)
        starts = np.unique(np.linspace(0, max_start, clips, dtype=int))
        for start in starts:
            indices = np.minimum(start + np.arange(num_frames) * stride, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            frames = []
            current = int(start)
            last = None
            for idx in indices:
                if idx < current and last is not None:  # 极短视频用最后一帧补齐。
                    frames.append(last)
                    continue
                while current < idx:
                    if not cap.grab():
                        raise RuntimeError(f"Decode failed at frame {current}: {path}")
                    current += 1
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at frame {idx}: {path}")
                current += 1
                last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(last)
            yield np.stack(frames), (indices / fps).tolist()
    finally:
        cap.release()


def log_prediction(mode, result):
    LOG.info("[%s] top3=%s | Normal=%.8f | rank=%d/14 | early_return=%s", mode,
             ", ".join(f"{label}:{prob:.6f}" for label, prob in result["top_k"][:3]),
             result["normal_probability"], result["normal_rank"], result["normal_early_return"])
    LOG.info("[%s] all14=%s", mode,
             json.dumps(result["all_probs"], ensure_ascii=False))


def summarize(report):
    import numpy as np
    summary = {}
    for mode in report["modes"]:
        rows = [v["results"][mode] for v in report["videos"] if mode in v["results"]]
        n = len(rows)
        if n:
            summary[mode] = {
                "evaluated": n,
                "normal_top1_count": sum(r["normal_in_top1"] for r in rows),
                "normal_top3_count": sum(r["normal_in_top3"] for r in rows),
                "normal_top1_recall": sum(r["normal_in_top1"] for r in rows) / n,
                "normal_top3_recall": sum(r["normal_in_top3"] for r in rows) / n,
                "normal_early_return_count": sum(r["normal_early_return"] for r in rows),
                "mean_normal_probability": float(np.mean([r["normal_probability"] for r in rows])),
                "median_normal_rank": float(np.median([r["normal_rank"] for r in rows])),
            }
    report["summary"] = summary


def save_report(report, directory):
    summarize(report)
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "results.json")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    paths, selection = select_videos(args)
    output = args.output_dir or ROOT / "experiments" / datetime.now().strftime("videomae_anomaly_%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(output / "console.log", encoding="utf-8")])
    LOG.info("Output: %s", output)
    # 默认只使用本机缓存；不会调用收费 API，也不会自动下载大权重。
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from video_anomaly_detection.config import PreprocessConfig
    from video_anomaly_detection.video_preprocessor import VideoPreprocessor
    from video_anomaly_detection.videomae_classifier import VideoMAEClassifier

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请改用 --device cpu")
    config_path = Path(hf_hub_download(args.model, "config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_labels = {int(i): label for i, label in config["id2label"].items()}
    # 明确核对原始权重里的 Normal ID；仅允许已有的 Normal_Videos_event 别名。
    normal_ids = [i for i, label in raw_labels.items() if label.lower().startswith("normal")]
    if normal_ids != [7] or len(raw_labels) != 14:
        raise ValueError(f"权重标签与当前分类器不匹配：{raw_labels}")
    for i, label in raw_labels.items():
        expected = "Normal" if i == 7 else label
        if VideoMAEClassifier.ID2LABEL[i] != expected:
            raise ValueError(f"标签 ID {i} 不匹配")
    LOG.info("Labels: %s", json.dumps(raw_labels, ensure_ascii=False))
    LOG.info("Normal id=7, raw_label=%s; device=%s; dtype=float32", raw_labels[7], device)
    preproc_config = PreprocessConfig()
    classifier = VideoMAEClassifier(model_name=args.model, device=device,
                                    num_frames=int(config.get("num_frames", 16)))
    alternate = None
    processor_config = None
    modes = ["pipeline_tta"]
    if not args.baseline_only:
        from transformers import VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained(args.model)
        processor_config = processor.to_dict()
        alternate = processor_classifier(classifier, processor)
        modes += ["processor_tta", "dense_processor"]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "selection": selection,
              "model": args.model, "revision": config_path.parent.name, "raw_id2label": raw_labels,
              "device": device, "dtype": "float32", "torch": torch.__version__,
              "transformers": transformers.__version__, "preprocess_config": asdict(preproc_config),
              "processor_config": processor_config, "modes": modes,
              "dense_stride_native_frames": args.dense_stride, "dense_clips": args.dense_clips,
              "classifier_source_sha256": hashlib.sha256((ROOT / "video_anomaly_detection/videomae_classifier.py").read_bytes()).hexdigest(),
              "note": "Normal-only diagnostic sample, not full-pipeline FPR or all-dataset accuracy.",
              "videos": []}
    save_report(report, output)
    for i, path in enumerate(paths, 1):
        LOG.info("VIDEO %d/%d %s", i, len(paths), path.name)
        record = {"video_path": str(path), "expected_category": "Normal", "results": {}}
        report["videos"].append(record)
        try:
            data = VideoPreprocessor(**asdict(preproc_config)).process(str(path))
            frames = data["all_frames"]
            if not len(frames):
                raise ValueError("No decoded frames")
            record.update(duration_seconds=data["duration"], fps=data["video_fps"],
                          retained_frame_count=len(frames),
                          tta_view_timestamps=tta_timestamps(data["timestamps"], classifier.num_frames))
            result = infer(classifier, frames, tta=True)
            record["results"]["pipeline_tta"] = result
            log_prediction("pipeline_tta", result)
            if alternate is not None:
                result = infer(alternate, frames, tta=True)
                record["results"]["processor_tta"] = result
                log_prediction("processor_tta", result)
                clips = []
                for clip, timestamps in dense_inputs(path, classifier.num_frames, args.dense_clips, args.dense_stride):
                    result = infer(alternate, clip, tta=False)
                    result["timestamps"] = timestamps
                    clips.append(result)
                    LOG.info("[dense clip %d] %.2f-%.2fs top1=%s Normal=%.8f rank=%d",
                             len(clips), timestamps[0], timestamps[-1], result["category"],
                             result["normal_probability"], result["normal_rank"])
                logits = torch.tensor([c["forward_logits"][0] for c in clips]).mean(dim=0)
                probs = torch.softmax(logits, dim=-1).tolist()
                ranked = np.argsort(probs)[::-1]
                result = describe_prediction({
                    "category": classifier.ID2LABEL[int(ranked[0])], "confidence": probs[int(ranked[0])],
                    "top_k": [(classifier.ID2LABEL[int(j)], probs[int(j)]) for j in ranked],
                    "all_probs": {classifier.ID2LABEL[j]: probs[j] for j in range(14)},
                    "aggregation": "mean logits across dense clips, then softmax", "clips": clips})
                record["results"]["dense_processor"] = result
                log_prediction("dense_processor", result)
            del frames, data
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            LOG.exception("Video failed: %s", path)
        save_report(report, output)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_report(report, output)
    LOG.info("SUMMARY %s", json.dumps(report["summary"], ensure_ascii=False))
    LOG.info("Saved: %s", output / "results.json")
    return 1 if any("error" in v for v in report["videos"]) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; completed videos remain in results.json")
        sys.exit(130)
    except Exception:
        LOG.exception("Test failed")
        sys.exit(1)
