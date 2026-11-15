"""XD test index and typed temporal ground truth, isolated from inference.

Filename labels are used ONLY to select/evaluate videos. They must never be
included in a VLM prompt or passed as a prediction prior.
"""
from pathlib import Path
import hashlib
import json
import random
try:
    from .xd_config import XD_CLASSES, XD_LABELS
except ImportError:
    from xd_config import XD_CLASSES, XD_LABELS


def default_annotations():
    return Path(__file__).resolve().parents[2] / 'data/xd_violence/annotations_multiclasses.txt'


def annotation_paths(root, annotations=None):
    local=Path(root)/'data/video/annotations_multiclasses.txt'
    typed=Path(annotations) if annotations else (local if local.exists() else default_annotations())
    binary=Path(root)/'data/video/test_annotations.txt'
    if not binary.exists():
        binary=default_annotations().with_name('annotations_binary.txt')
    return typed,binary


def parse_annotations(path, typed=True):
    result = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name, values = parts[0], parts[1:]
        # Official files mix names with and without .mp4 on fourteen rows.
        # Strip only a known video suffix; movie titles themselves contain dots.
        if Path(name).suffix.lower() in ('.mp4','.avi','.mkv','.mov'):
            name=Path(name).stem
        if name in result or len(values) % (3 if typed else 2):
            raise ValueError('Duplicate/malformed annotation: ' + name)
        events = []
        for i in range(0, len(values), 3 if typed else 2):
            category = XD_LABELS[values[i]] if typed else None
            start, end = map(int, values[i + 1:i + 3] if typed else values[i:i + 2])
            if not 0 <= start < end or category == 'Normal':
                raise ValueError('Invalid abnormal event: ' + name)
            events.append(dict(category=category, start_frame=start, end_frame=end))
        result[name] = events
    return result


def load_xd_index(root, annotations=None):
    root = Path(root)
    typed_path,binary_path = annotation_paths(root,annotations)
    typed = parse_annotations(typed_path)
    binary = parse_annotations(binary_path, typed=False)
    if set(typed) != set(binary):
        raise ValueError('Binary and class-specific annotation coverage differ')
    paths = sorted(p for p in (root / 'data/video/test_videos').rglob('*')
                   if p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov'))
    records = {}
    for path in paths:
        name = path.stem
        if name in records or '_label_' not in name:
            raise ValueError('Ambiguous XD video identity: ' + name)
        codes = name.rsplit('_label_', 1)[1].split('-')
        labels = [XD_LABELS[c] for c in codes if c != '0']
        normal = labels == ['Normal']
        if not normal and name not in typed:
            raise ValueError('Missing abnormal annotation: ' + name)
        if normal and name in typed:
            raise ValueError('Normal video has abnormal annotations: ' + name)
        records[name] = dict(video_id=name, video_path=str(path.resolve()),
            video_labels=labels, category='Normal' if normal else labels[0],
            events=typed.get(name, []),
            segments=[(e['start_frame'], e['end_frame']) for e in binary.get(name, [])])
    if set(typed) - set(records):
        raise ValueError('Annotated videos missing from disk')
    return records


def select_xd_videos(root, per_class=5, annotations=None, seed=42, categories=None):
    """Five designated samples/class; overlap is diagnosed once and disclosed."""
    import pandas as pd
    records = load_xd_index(root, annotations)
    classes = list(categories or (*XD_CLASSES, 'Normal'))
    if per_class < 1 or set(classes) - set((*XD_CLASSES, 'Normal')):
        raise ValueError('Invalid sample count/categories')
    rng = random.Random(seed)
    selected, slots = set(), {}
    for category in classes:
        # Use actual per-event labels for anomaly evaluation, not just the first
        # filename label. Choose without replacement within each category.
        pool = sorted(name for name, row in records.items() if
                      (category == 'Normal' and row['category'] == 'Normal') or
                      any(e['category'] == category for e in row['events']))
        if len(pool) < per_class:
            raise ValueError(f'Insufficient videos for {category}: {len(pool)}')
        slots[category] = sorted(rng.sample(pool, per_class))
        selected.update(slots[category])
    rows = [dict(records[name], category=','.join(records[name]['video_labels'])) for name in sorted(selected)]
    df = pd.DataFrame(rows)
    df.attrs['selection'] = dict(seed=seed, per_class=per_class, slots=slots,
        unique_video_count=len(rows), actual_event_class_coverage={c:sum(
            any(e['category']==c for e in records[n]['events']) if c!='Normal'
            else records[n]['category']=='Normal' for n in selected) for c in classes})
    return df


def annotation_fingerprint(root, annotations=None):
    paths = annotation_paths(root,annotations)
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
