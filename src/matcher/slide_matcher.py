import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os


class SlideMatcher:
    def __init__(self, video_path: str, slides_dir: str):
        """
        初始化幻灯片匹配器

        Args:
            video_path: 视频文件路径
            slides_dir: 幻灯片图片目录
        """
        self.video_path = video_path
        self.slides_dir = slides_dir
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载所有幻灯片
        self.slides = self._load_slides()

    def _load_slides(self) -> List[Dict[str, Any]]:
        """加载幻灯片并提取特征"""
        slides = []
        slide_files = [
            f
            for f in os.listdir(self.slides_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for slide_file in slide_files:
            slide_path = os.path.join(self.slides_dir, slide_file)
            slide = cv2.imread(slide_path)
            if slide is not None:
                # 调整大小以匹配视频分辨率
                slide = cv2.resize(slide, (self.width, self.height))
                # 提取特征
                features = self._extract_features(slide)
                slides.append(
                    {
                        "path": slide_path,
                        "name": slide_file,
                        "image": slide,
                        "features": features,
                    }
                )

        return slides

    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取图像特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 文本检测
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 2. 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # 3. 提取SIFT特征
        sift = cv2.SIFT_create(nfeatures=1000)  # 增加特征点数量
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # 4. 计算文本分布
        rows, cols = 3, 3  # 使用3x3网格
        cell_height = text_mask.shape[0] // rows
        cell_width = text_mask.shape[1] // cols
        text_distribution = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                cell = text_mask[
                    i * cell_height : (i + 1) * cell_height,
                    j * cell_width : (j + 1) * cell_width,
                ]
                text_distribution[i, j] = np.sum(cell > 0) / cell.size

        return {
            "histogram": hist,
            "descriptors": descriptors if descriptors is not None else np.array([]),
            "keypoints": keypoints,
            "text_mask": text_mask,
            "text_distribution": text_distribution.flatten(),
        }

    def match_frame(
        self, frame: np.ndarray, min_similarity: float = 0.3  # 进一步降低阈值
    ) -> Optional[Dict[str, Any]]:
        """匹配视频帧与幻灯片"""
        frame_features = self._extract_features(frame)
        best_match = None
        best_score = 0
        scores = []  # 用于调试输出

        for slide in self.slides:
            score = self._compute_similarity(frame_features, slide["features"])
            scores.append((slide["name"], score))
            if score > best_score:
                best_score = score
                best_match = slide

        # 输出前3个最佳匹配的分数
        scores.sort(key=lambda x: x[1], reverse=True)
        print("\n最佳匹配分数:")
        for name, score in scores[:3]:
            print(f"- {name}: {score:.3f}")

        if best_score >= min_similarity:
            return best_match
        return None

    def _compute_similarity(
        self, features1: Dict[str, Any], features2: Dict[str, Any]
    ) -> float:
        """计算两组特征的相似度"""
        # 1. 计算直方图相似度
        hist_score = cv2.compareHist(
            features1["histogram"].reshape(-1, 1),
            features2["histogram"].reshape(-1, 1),
            cv2.HISTCMP_CORREL,
        )

        # 2. 计算SIFT特征匹配
        if len(features1["descriptors"]) > 0 and len(features2["descriptors"]) > 0:
            bf = cv2.BFMatcher()
            try:
                matches = bf.knnMatch(
                    features1["descriptors"], features2["descriptors"], k=2
                )
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.85 * n.distance:  # 放宽比率测试
                        good_matches.append(m)
                sift_score = len(good_matches) / max(
                    len(features1["keypoints"]), len(features2["keypoints"])
                )
            except:
                sift_score = 0
        else:
            sift_score = 0

        # 3. 计算文本分布相似度
        distribution_score = 1 - np.mean(
            np.abs(features1["text_distribution"] - features2["text_distribution"])
        )

        # 4. 计算文本区域的IoU
        intersection = np.logical_and(features1["text_mask"], features2["text_mask"])
        union = np.logical_or(features1["text_mask"], features2["text_mask"])
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

        # 调整权重
        weights = {
            "hist": 0.3,  # 增加直方图权重
            "sift": 0.3,  # 增加SIFT权重
            "distribution": 0.2,
            "iou": 0.2,
        }

        final_score = (
            weights["hist"] * max(0, hist_score)
            + weights["sift"] * sift_score
            + weights["distribution"] * distribution_score
            + weights["iou"] * iou_score
        )

        return final_score

    def match_video_segment(
        self, start_frame: int, end_frame: int
    ) -> Optional[Dict[str, Any]]:
        """
        匹配视频片段

        Args:
            start_frame: 起始帧
            end_frame: 结束帧

        Returns:
            匹配的幻灯片信息，如果没有匹配则返回None
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 计算采样点数量（根据片段长度调整）
        segment_length = end_frame - start_frame
        num_samples = min(5, max(1, segment_length // 30))  # 每30帧最多采样一次

        # 采样该片段的几个关键帧进行匹配
        sample_points = np.linspace(start_frame, end_frame, num=num_samples, dtype=int)

        matches = []
        for frame_idx in sample_points:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue

            match = self.match_frame(frame)
            if match is not None:
                matches.append(match)

        if not matches:
            return None

        # 返回出现次数最多的匹配
        from collections import Counter

        match_counts = Counter(match["name"] for match in matches)
        best_match_name = match_counts.most_common(1)[0][0]
        best_match = next(
            slide for slide in matches if slide["name"] == best_match_name
        )

        return best_match

    def match_timestamps(
        self, timestamps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        为每个时间戳匹配对应的幻灯片

        Args:
            timestamps: 时间戳列表

        Returns:
            带有匹配信息的时间戳列表
        """
        matched_timestamps = []
        total = len(timestamps)

        print(f"\n开始匹配 {len(self.slides)} 张幻灯片...")
        for i, ts in enumerate(timestamps, 1):
            print(f"\r正在处理第 {i}/{total} 个时间点...", end="", flush=True)

            start_frame = int(ts["time"].total_seconds() * self.fps)
            if i < len(timestamps):
                end_frame = int(timestamps[i]["time"].total_seconds() * self.fps)
            else:
                end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            match = self.match_video_segment(start_frame, end_frame)
            matched_ts = ts.copy()
            if match is not None:
                matched_ts.update(
                    {"matched_slide": match["name"], "slide_path": match["path"]}
                )
                print(f"\n找到匹配: {match['name']}")
            else:
                print("\n未找到匹配的幻灯片")
            matched_timestamps.append(matched_ts)

        print("\n匹配完成!")
        return matched_timestamps

    def __del__(self):
        """清理资源"""
        if hasattr(self, "cap"):
            self.cap.release()
