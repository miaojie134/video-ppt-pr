import cv2
import numpy as np
from collections import deque


class MotionHistory:
    def __init__(self, max_size=10):
        self.motion_history = deque(maxlen=max_size)
        self.activity_map = None
        self.activity_threshold = 0.3

    def update(self, motion_vectors, frame_shape):
        if self.activity_map is None:
            self.activity_map = np.zeros(frame_shape, dtype=np.float32)

        motion_map = np.zeros(frame_shape, dtype=np.float32)
        if len(motion_vectors) > 0:
            points = motion_vectors.reshape(-1, 2, 2)  # 重塑为 (N, 2, 2) 形状
            for point_pair in points:
                start_point = point_pair[0]  # 起始点
                x, y = int(start_point[0]), int(start_point[1])
                if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                    motion_map[y, x] = 1

        self.motion_history.append(motion_map)

        # 优化平均值计算，避免空数组警告
        if len(self.motion_history) > 0:
            history_array = np.array(self.motion_history)
            if history_array.size > 0:
                self.activity_map = np.mean(history_array, axis=0)
            else:
                self.activity_map = np.zeros(frame_shape, dtype=np.float32)
        else:
            self.activity_map = np.zeros(frame_shape, dtype=np.float32)

    def get_active_regions(self):
        if self.activity_map is None:
            return None
        # 确保返回布尔数组
        active_regions = self.activity_map > self.activity_threshold
        return active_regions.astype(np.uint8)


def detect_lecturer_region(frame_gray, motion_history):
    """动态检测讲师区域"""
    if motion_history.activity_map is None:
        # 首次运行时，使用默认分割
        width = frame_gray.shape[1]
        lecturer_mask = np.zeros(frame_gray.shape, dtype=np.uint8)
        lecturer_mask[:, int(width * 0.7) :] = 1
        return lecturer_mask

    active_regions = motion_history.get_active_regions()
    if active_regions is None:
        return None

    # 对活动区域进行形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    active_regions = cv2.morphologyEx(
        active_regions.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    )

    return active_regions


def analyze_optical_flow(prev_gray, curr_gray, motion_history=None):
    """使用改进的Lucas-Kanade方法分析两帧之间的光流特征"""
    if motion_history is None:
        motion_history = MotionHistory()

    # 检测讲师区域
    lecturer_mask = detect_lecturer_region(prev_gray, motion_history)

    # 特征点检测参数
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    # 在整个画面检测特征点
    points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if points is None:
        return {
            "vertical_flow": 0,
            "horizontal_flow": 0,
            "mean_magnitude": 0,
            "max_magnitude": 0,
            "motion_consistency": 0,
            "motion_coherence": 0,
            "is_gradual": False,
            "is_lecturer_motion": False,
            "lecturer_movement_ratio": 0,
        }

    # Lucas-Kanade参数
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # 计算光流
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None, **lk_params
    )

    if next_points is None:
        return {
            "vertical_flow": 0,
            "horizontal_flow": 0,
            "mean_magnitude": 0,
            "max_magnitude": 0,
            "motion_consistency": 0,
            "motion_coherence": 0,
            "is_gradual": False,
            "is_lecturer_motion": False,
            "lecturer_movement_ratio": 0,
        }

    # 分离有效点
    good_points = points[status == 1]
    good_next_points = next_points[status == 1]

    if len(good_points) == 0:
        return {
            "vertical_flow": 0,
            "horizontal_flow": 0,
            "mean_magnitude": 0,
            "max_magnitude": 0,
            "motion_consistency": 0,
            "motion_coherence": 0,
            "is_gradual": False,
            "is_lecturer_motion": False,
            "lecturer_movement_ratio": 0,
        }

    # 计算运动向量
    motion_vectors = np.hstack((good_points, good_next_points))
    flow_vectors = good_next_points - good_points

    # 更新运动历史
    motion_history.update(motion_vectors, prev_gray.shape)

    # 分离讲师区域和PPT区域的运动
    lecturer_points_mask = np.array(
        [lecturer_mask[int(p[1]), int(p[0])] > 0 for p in good_points]
    )

    ppt_vectors = flow_vectors[~lecturer_points_mask]
    lecturer_vectors = flow_vectors[lecturer_points_mask]

    # 分析PPT区域的运动
    if len(ppt_vectors) > 0:
        ppt_magnitudes = np.sqrt(ppt_vectors[:, 0] ** 2 + ppt_vectors[:, 1] ** 2)
        ppt_angles = np.arctan2(ppt_vectors[:, 1], ppt_vectors[:, 0])

        mean_magnitude = float(np.mean(ppt_magnitudes))
        max_magnitude = float(np.max(ppt_magnitudes))

        # 计算运动一致性
        angles_hist = np.histogram(ppt_angles, bins=8, range=(-np.pi, np.pi))[0]
        motion_consistency = float(np.max(angles_hist) / len(ppt_angles))

        # 计算运动连贯性
        magnitude_std = np.std(ppt_magnitudes)
        motion_coherence = 1.0 - (magnitude_std / (mean_magnitude + 1e-6))

        # 计算主要运动方向
        vertical_flow = float(np.mean([abs(v[1]) for v in ppt_vectors]))
        horizontal_flow = float(np.mean([abs(v[0]) for v in ppt_vectors]))
    else:
        mean_magnitude = 0
        max_magnitude = 0
        motion_consistency = 0
        motion_coherence = 0
        vertical_flow = 0
        horizontal_flow = 0

    # 分析讲师动作特征
    lecturer_movement_ratio = 0
    if len(lecturer_vectors) > 0:
        lecturer_magnitudes = np.sqrt(
            lecturer_vectors[:, 0] ** 2 + lecturer_vectors[:, 1] ** 2
        )
        significant_motion = np.sum(lecturer_magnitudes > 0.5)
        lecturer_movement_ratio = significant_motion / len(lecturer_vectors)

    # 改进的讲师动作判断
    is_lecturer_motion = (
        lecturer_movement_ratio > 0.3  # 讲师区域有显著运动
        or (  # 或者PPT区域的运动符合讲师动作特征
            len(ppt_vectors) > 0
            and (
                (motion_coherence < 0.4 and mean_magnitude < 1.5)  # 不连贯的小幅度运动
                or (
                    mean_magnitude < 0.8  # 运动幅度较小
                    and motion_consistency < 0.3  # 运动方向不一致
                    and len(lecturer_vectors) > 0  # 同时讲师区域有运动
                )
            )
        )
    )

    # 判断是否为渐变
    is_gradual = (
        not is_lecturer_motion
        and len(ppt_vectors) > 0
        and motion_consistency > 0.4
        and motion_coherence > 0.6
        and mean_magnitude > 0.5
    )

    return {
        "vertical_flow": vertical_flow,
        "horizontal_flow": horizontal_flow,
        "mean_magnitude": mean_magnitude,
        "max_magnitude": max_magnitude,
        "motion_consistency": motion_consistency,
        "motion_coherence": motion_coherence,
        "is_gradual": bool(is_gradual),
        "is_lecturer_motion": bool(is_lecturer_motion),
        "lecturer_movement_ratio": float(lecturer_movement_ratio),
    }
