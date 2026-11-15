# XD-Violence support

The `--dataset xd_violence` route uses the XD VideoMAE checkpoint for a local-window
multi-label prior, CLIP for proposals/boundary refinement, and a VLM to verify
individual events. UCF remains the default route and retains its own checkpoint,
category assignments and postprocessing.

## Run the five-per-category evaluation

Set `DASHSCOPE_API_KEY` in the current process using your existing credential.
The program never prints it. From the repository root:

```powershell
conda run --no-capture-output -n common python vad_v3/video_anomaly_detection/batch_evaluate.py `
  --dataset xd_violence `
  --dataset_dir F:/dataset/xd-violence `
  --output_dir vad_v3/experiments/v218 `
  --max_per_category 5 --seed 42 `
  --device cuda --clip_model ViT-L-14 `
  --concurrency 2 --api_concurrency 4 `
  --detect_model qwen2.5-vl-72b-instruct `
  --classify_model qwen3-vl-plus `
  --max_tokens 2048 --max_frames 32 `
  --api_timeout_seconds 120 --mcts_mode off
```

An existing output directory is resumable only with the same source/config/input
fingerprint. Choose a new experiment directory after changing code or parameters.
With MCTS off, only the classification VLM is called. Detection/MCTS has a
separate client for a later experiment; its model's availability is not established
by an off-mode run.

Selection designates five videos for each of the six anomaly classes and Normal.
Anomaly membership uses typed event annotations. A multi-label video is diagnosed
only once; incidental coverage may exceed five for some classes. The exact slots,
seed and actual coverage are in `metadata/selection.json`.

Single-video inference is also available through `run.py --dataset xd_violence`.
It uses the same factory route. `VADConfig.xd` contains the window, evidence-frame,
boundary and postprocessing settings; `run.py --config_json` can load a saved
configuration.

## Model and preprocessing provenance

- Checkpoint: `KingTechnician/videomae-small-finetuned-kinetics-finetuned-xd-violence-multilabel`
- Revision: `54875f0b60e8acad7f1f8934e3f20ca6118f5b36`
- Class IDs: `0 Normal`, `1 Fighting`, `2 Shooting`, `3 Riot`, `4 Abuse`,
  `5 CarAccident`, `6 Explosion`.
- Multi-label logits use **sigmoid**, not softmax or a single winning label.
- The checkpoint has **no processor configuration**. This implementation explicitly
  assumes its base checkpoint's ImageNet-normalized, resized/center-cropped
  processor: `MCG-NJU/videomae-small-finetuned-kinetics`, revision
  `1bf2b82c809609f5df1ea7648aee577e2e486b63`.
- This preprocessing and the eight-second window/four-second stride are experimental
  choices, not a verified reproduction of the author's fine-tuning recipe.
- Each window samples 16 frames by timestamps. Video-level evidence is the mean of
  the top three window scores/class; all window scores are retained in the result.

The frame pool disables motion deletion and resizes the longer edge to at most
448 pixels to bound memory. The requested frame sampling rate defaults to 2 FPS;
actual source-frame timestamps, not an assumed exact rate, are retained.

## Inference semantics

1. Shared CLIP image features generate all six class-versus-Normal curves. Riot
   has explicit descriptions and the normal prompt set includes peaceful crowds.
   UCF learned prompt weights and MIL weights are not loaded on this route.
2. Each class contributes up to two local inspection windows, scored using its
   local VideoMAE and CLIP evidence. Two uniform coverage windows are added.
   Identical windows share a VLM call. This is a bounded proposal scan, not an
   exhaustive inspection of every instant of a long video.
3. A window's VLM request includes the full six-class vocabulary and explicit
   image-ID/timestamp mapping. Neither filename labels nor numeric model priors
   are given to the VLM. Each event has its own class, boundaries, confidence and
   supporting image IDs. Events extending beyond the inspected window are clipped
   to its intersection; disjoint and zero-length intervals remain invalid.
   Invalid replies get up to two schema-repair requests with the specific error
   and previous response; exhausted
   failures are not converted to Normal results. Failed response evidence is saved
   under `metadata/failed_responses`.
4. Local CLIP refinement can move each boundary by at most 20% of event duration.
   Flat curves preserve the VLM interval. There are no UCF category duration caps,
   forced three-second event minima, whole-video Normal exits or class unification.
5. Close, mostly nonoverlapping fragments merge only within a class. NMS at 0.5
   also runs within each class, preserving simultaneous different categories.

The prior and VLM confidence are not calibrated probabilities. All six classes
receive a discovery path even if their video-level prior rank is low.

Optional MCTS uses each event's own category and curve, with a shared total
per-video request/candidate budget across classes. Search rewards never overwrite
the detection confidence. `--mcts_paired_eval` can retain an `original_off` arm
when used with `--mcts_mode refine`; the UCF-specific Stage-4 ablations and boundary
guard are rejected on XD. The initial v218 run has MCTS **off** and does not measure
its benefit.

## Data and evaluation

Videos are read from `F:/dataset/xd-violence/data/video/test_videos`.
The local `annotations_multiclasses.txt` is preferred for per-event class GT;
`--xd_annotations` can override it. A cached version exists under
`data/xd_violence`, pinned to VadCLIP commit
`c41067f07d252efcda18008bea367886070c33b0`.

Binary GT uses the official `test_annotations.txt` if present, otherwise the
downloaded `data/xd_violence/annotations_binary.txt`. Its URL and SHA-256 are stored
alongside it. Typed and binary annotation coverage must match. Fourteen official
rows include `.mp4` while other rows omit it; the loader normalizes only known
video extensions, preserving dots in movie titles.

- **frame_AP**: `average_precision_score` over concatenated selected-video frames,
  using the original binary GT and timestamp-interpolated predictions.
- **frame_AUC**: a supplemental binary metric.
- **temporal_mAP_abnormal**: six anomaly classes, abnormal selected videos only.
- **temporal_mAP_full**: the same six classes, including selected Normal videos so
  their anomalous predictions count as false positives. Normal is not a seventh
  class in the temporal macro average.
- Thresholds: tIoU 0.1, 0.2, 0.3, 0.4, 0.5 and AVG. Matching is by video and class,
  descending confidence, one GT per TP. Duplicate detections count as FPs.
- AP formula: `sum(precision_at_rank * TP) / N_GT`, without VOC interpolation.

`metadata/video_fps.json` records each evaluated video's native OpenCV FPS and
frame count. Frame intervals use `[start,end)`. Variable frame rate is not verified;
the initial selected 35 videos all report 24 FPS. This is a small diagnostic subset,
not a comparable full 800-video benchmark result.

Results retain priors, raw VLM replies, image timestamps, intermediate/final events
and token usage. `metadata/temporal_matching_audit.json` contains ranked TP/FP
matches. Per-call usage remains in structured JSONL, not verbose console lines.

Primary annotation sources:

- https://roc-ng.github.io/XD-Violence/images/annotations.txt
- https://github.com/nwpu-zxr/VadCLIP/blob/c41067f07d252efcda18008bea367886070c33b0/list/annotations_multiclasses.txt

## Checks

`tests/test_xd_support.py` covers overlapping classes, duplicate detections,
low-prior Riot coverage, invalid responses versus empty successful responses,
native-FPS matching, Normal false positives and annotation/config parsing.
Existing UCF config, result-schema and postprocessor tests are also run.
