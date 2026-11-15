"""Optional independent dense frame pool. Never mutates upstream video arrays."""
import time
import cv2
import numpy as np


# 创建独立的局部帧池和CLIP曲线，保留全局video_data供原流水线及off分支使用。
# 此预处理在单候选搜索计时前执行，解码/编码耗时单独记录。
def local_dense_pool(video_data, seeds, category, frame_scorer, cfg):
    windows=[]
    for seed in seeds.values():
        s,e=seed['t_start'],seed['t_end']; pad=(e-s)*cfg.search_padding_ratio+cfg.context_seconds
        windows.append((max(0,s-pad),min(video_data['duration'],e+pad)))
    # 只采搜索窗口及背景上下文；重叠窗口去重，超预算报错而非悄悄降低密度。
    targets=sorted(set(float(t) for s,e in windows for t in np.arange(s,e,1/cfg.dense_fps)))
    if len(targets)>cfg.max_dense_frames:
        raise ValueError('dense_frame_budget')
    started=time.monotonic(); cap=cv2.VideoCapture(video_data['video_path'])
    frames=[]; times=[]
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps<=0: raise ValueError('dense_decode_unavailable')
        for index in sorted(set(round(t*fps) for t in targets)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok: raise ValueError('dense_decode_failed')
            # 使用实际解码位置对应时间，不能直接把请求的目标时间当作真实帧时间。
            actual=(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)/fps
            if actual<0 or (times and actual<=times[-1]): raise ValueError('dense_timestamp_invalid')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); times.append(actual)
    finally:
        cap.release()
    pool={'video_path':video_data['video_path'],'duration':video_data['duration'],
          'all_frames':np.asarray(frames),'timestamps':np.asarray(times),'evidence_pool':'dense',
          'decode_seconds':time.monotonic()-started}
    started=time.monotonic()
    pool['clip_features']=frame_scorer.anomaly_expert.precompute_video_features(pool['all_frames'])
    pool['encode_seconds']=time.monotonic()-started
    curve=frame_scorer.get_category_scores(pool,category)
    if curve is None: raise ValueError('dense_curve_unavailable')
    return pool,curve
