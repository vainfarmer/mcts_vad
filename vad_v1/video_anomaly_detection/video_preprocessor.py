"""
视频预处理与初始分段模块

Pipeline Stage 1 的核心组件，负责将原始视频转化为后续所有阶段的输入数据结构。

两步处理流程：
  Step 1: 按固定帧率采样（fps），得到 all_frames + timestamps
          附带可选的运动过滤（motion_threshold），跳过无运动帧以减少 CLIP 编码量
  Step 2: 按固定时长（segment_duration）切分为初始段列表 segments
          支持可选的 50% 重叠滑动窗口，防止异常在段边界处被切割漏检
"""

import bisect
import numpy as np
import cv2


class VideoPreprocessor:
    """
    视频预处理器：抽帧 + 运动过滤 + 分段（支持重叠）
    """

    def __init__(self, fps=4, segment_duration=6,
                 motion_threshold=1.0, min_motion_interval=2.0,
                 segment_overlap=0.5):
        """
        Args:
            fps: 采样帧率
            segment_duration: 初始分段长度（秒）
            motion_threshold: 帧差均值阈值，设为 0 禁用运动过滤
            min_motion_interval: 静态区域最低保留间隔（秒）
            segment_overlap: 段重叠比例（0=无重叠，0.5=50%重叠滑动窗口）
        """
        self.fps = fps
        self.segment_duration = segment_duration
        self.motion_threshold = motion_threshold
        self.min_motion_interval = min_motion_interval
        self.segment_overlap = segment_overlap

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid video FPS ({video_fps}): {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        sample_interval = max(1, int(video_fps / self.fps))

        frames = []
        timestamps = []
        idx = 0
        while True:
            if idx % sample_interval == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(idx / video_fps)
            else:
                if not cap.grab():
                    break
            idx += 1
        cap.release()

        frames = np.array(frames)
        timestamps = np.array(timestamps)

        if self.motion_threshold > 0 and len(frames) > 1:
            small = np.stack([cv2.resize(f, (64, 64)) for f in frames])
            motion_scores = np.zeros(len(frames))
            diffs = np.abs(small[1:].astype(np.float32) - small[:-1].astype(np.float32))
            motion_scores[1:] = diffs.mean(axis=(1, 2, 3))

            keep = np.zeros(len(frames), dtype=bool)
            keep[0] = True
            last_kept_time = timestamps[0]
            for i in range(1, len(frames)):
                if motion_scores[i] > self.motion_threshold:
                    keep[i] = True
                    last_kept_time = timestamps[i]
                elif timestamps[i] - last_kept_time >= self.min_motion_interval:
                    keep[i] = True
                    last_kept_time = timestamps[i]

            n_before = len(frames)
            frames = frames[keep]
            timestamps = timestamps[keep]
            print(f"  [Motion filter] {n_before} -> {len(frames)} frames "
                  f"({100*(1-len(frames)/n_before):.0f}% removed, "
                  f"threshold={self.motion_threshold})")

        # 分段：支持重叠滑动窗口
        stride = self.segment_duration * (1.0 - self.segment_overlap)
        segments = []
        seg_start = 0.0
        while seg_start < duration:
            seg_end = min(seg_start + self.segment_duration, duration)
            # 避免尾部产生过短的段
            if seg_end - seg_start < self.segment_duration * 0.3 and segments:
                break
            lo = bisect.bisect_left(timestamps, seg_start)
            hi = bisect.bisect_left(timestamps, seg_end)
            seg_frame_mask = list(range(lo, hi))
            if seg_frame_mask:
                segments.append({
                    't_start': float(seg_start),
                    't_end': float(seg_end),
                    'frame_indices': seg_frame_mask,
                })
            seg_start += stride

        return {
            'video_path': video_path,
            'duration': duration,
            'all_frames': frames,
            'timestamps': timestamps,
            'segments': segments,
            'video_fps': video_fps,
        }

    @staticmethod
    def get_segment_frames(video_data, t_start, t_end):
        timestamps = video_data['timestamps']
        lo = bisect.bisect_left(timestamps, t_start)
        hi = bisect.bisect_left(timestamps, t_end)
        indices = list(range(lo, hi))
        if not indices:
            mid = (t_start + t_end) / 2
            closest = int(np.argmin(np.abs(timestamps - mid)))
            indices = [closest]
        return video_data['all_frames'][indices], indices
