"""Keep first-pass VLM evidence separate from refined detection boundaries."""
from copy import deepcopy


def discovery_descriptions(candidate):
    """Return verbatim descriptions with their ORIGINAL scan windows."""
    if 'original_descriptions' in candidate:
        return deepcopy(candidate['original_descriptions'])
    description = candidate.get('description')
    if description is None:
        return []
    return [{
        't_start': float(candidate['t_start']),
        't_end': float(candidate['t_end']),
        'description': description,
    }]


def attach_descriptions(anomalies, sources):
    """Resolve lineage after all boundary edits, merges and NMS are complete.

    Deliberately use candidate IDs, never time overlap: refined windows can move
    outside the discovery window, and an NMS-suppressed event is not a source.
    """
    for anomaly in anomalies:
        # 最终边界可能已经移动，必须按来源 ID 关联，不能按当前时间重叠猜测描述来源。
        records = []
        for source_id in anomaly.get('source_candidate_ids', []):
            for record in sources.get(source_id, []):
                if record not in records:
                    records.append(deepcopy(record))
        records.sort(key=lambda row: (row['t_start'], row['t_end']))
        if records:
            anomaly['original_descriptions'] = records
            anomaly['description'] = records[0]['description']
    return anomalies
