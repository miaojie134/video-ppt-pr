#!/usr/bin/env python3

import cv2
import numpy as np
from datetime import timedelta, datetime
import csv
import argparse
import os
import subprocess
import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow

# 全局默认参数
DEFAULT_THRESHOLD = 0.25
DEFAULT_MIN_INTERVAL = 0.3


def format_timedelta(td):
    """将timedelta转换为HH:MM:SS:FF格式（帧数）"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 假设25帧/秒，将毫秒转换为帧号(0-24)
    frames = int((td.microseconds / 1000000) * 25)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def analyze_frame(frame):
    """分析帧的特征，关注左侧内容区域"""
    height, width = frame.shape[:2]

    # 调整图像大小以提高性能
    frame = cv2.resize(frame, (960, 540))

    # 主要关注左侧2/3区域（通常是PPT内容区域）
    content_width = int(width * 0.67)  # 取左侧2/3作为内容区域
    content_region = frame[:, :content_width]

    # 转换为灰度图
    gray = cv2.cvtColor(content_region, cv2.COLOR_BGR2GRAY)

    # 1. 计算直方图 - 增加bin数量以提高精度
    hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 2. 计算亮度和对比度
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # 3. 文本检测（使用自适应阈值）
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 使用形态学操作提取文本区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 计算文本密度和分布
    rows, cols = 8, 8  # 增加网格数量以提高精度
    cell_height = text_mask.shape[0] // rows
    cell_width = text_mask.shape[1] // cols
    region_densities = []

    for i in range(rows):
        for j in range(cols):
            region = text_mask[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ]
            density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
            region_densities.append(density)

    text_density = np.mean(region_densities)
    text_density_variance = np.var(region_densities)

    # 5. 计算文本的垂直和水平分布
    vertical_splits = 30  # 增加垂直分割数
    horizontal_splits = 30  # 增加水平分割数

    v_height = text_mask.shape[0] // vertical_splits
    h_width = text_mask.shape[1] // horizontal_splits

    vertical_profile = []
    horizontal_profile = []

    for i in range(vertical_splits):
        section = text_mask[i * v_height : (i + 1) * v_height, :]
        vertical_profile.append(np.sum(section) / section.size)

    for i in range(horizontal_splits):
        section = text_mask[:, i * h_width : (i + 1) * h_width]
        horizontal_profile.append(np.sum(section) / section.size)

    vertical_diff = np.mean(np.abs(np.diff(vertical_profile)))
    horizontal_diff = np.mean(np.abs(np.diff(horizontal_profile)))

    # 6. 计算边缘特征
    edges = cv2.Canny(gray, 30, 150)  # 降低阈值以捕获更多边缘
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 7. 计算局部对比度
    kernel = np.ones((7, 7), np.float32) / 49  # 增加核大小
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_contrast = np.std(gray - local_mean)

    return (
        hist,
        text_density,
        gray,
        text_mask,
        edge_density,
        local_contrast,
        vertical_diff,
        horizontal_diff,
        text_density_variance,
        brightness,  # 新增
        contrast,  # 新增
    )


def analyze_optical_flow(prev_gray, curr_gray):
    """使用Lucas-Kanade方法分析两帧之间的光流特征，并支持渐变检测"""
    # 在图像中选取特征点
    feature_params = dict(
        maxCorners=200,  # 增加特征点数量以提高检测精度
        qualityLevel=0.2,  # 降低特征点质量要求以捕获更多特征
        minDistance=5,  # 减小特征点间距以捕获更细微的变化
        blockSize=5,  # 减小检测区域大小以提高局部敏感度
    )

    # 获取特征点
    prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if prev_points is None:
        return {
            "vertical_flow": 0,
            "horizontal_flow": 0,
            "mean_magnitude": 0,
            "max_magnitude": 0,
            "motion_consistency": 0,
            "motion_uniformity": 1.0,
            "is_gradual": False,
        }

    # Lucas-Kanade光流参数
    lk_params = dict(
        winSize=(15, 15),  # 搜索窗口大小
        maxLevel=2,  # 金字塔层数
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # 计算光流
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )

    # 选取好的特征点
    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    if len(good_prev) == 0:
        return {
            "vertical_flow": 0,
            "horizontal_flow": 0,
            "mean_magnitude": 0,
            "max_magnitude": 0,
            "motion_consistency": 0,
            "motion_uniformity": 1.0,
            "is_gradual": False,
        }

    # 计算移动向量
    flow_vectors = good_next - good_prev

    # 计算垂直和水平方向的平均运动
    vertical_flow = np.mean(np.abs(flow_vectors[:, 1]))
    horizontal_flow = np.mean(np.abs(flow_vectors[:, 0]))

    # 计算运动幅度和角度
    magnitude = np.sqrt(flow_vectors[:, 0] ** 2 + flow_vectors[:, 1] ** 2)
    angle = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])

    # 计算整体运动强度
    mean_magnitude = np.mean(magnitude)
    max_magnitude = np.max(magnitude)  # 添加最大幅度检测

    # 分析运动的一致性
    angles_hist = np.histogram(angle, bins=8, range=(-np.pi, np.pi))[0]
    dominant_motion = (
        np.max(angles_hist) / np.sum(angles_hist) if np.sum(angles_hist) > 0 else 0
    )

    # 计算运动的空间分布
    height, width = prev_gray.shape
    grid_size = 4
    cell_height = height // grid_size
    cell_width = width // grid_size
    motion_distribution = []

    for i in range(grid_size):
        for j in range(grid_size):
            mask = (
                (good_prev[:, 1] >= i * cell_height)
                & (good_prev[:, 1] < (i + 1) * cell_height)
                & (good_prev[:, 0] >= j * cell_width)
                & (good_prev[:, 0] < (j + 1) * cell_width)
            )
            if np.any(mask):
                cell_magnitude = np.mean(magnitude[mask])
                motion_distribution.append(cell_magnitude)

    motion_uniformity = np.std(motion_distribution) if motion_distribution else 1.0

    # 检测渐变特征
    # 1. 运动一致性高
    # 2. 运动幅度适中
    # 3. 特征点分布均匀
    is_gradual = (
        dominant_motion > 0.3  # 降低运动方向一致性要求
        and 0.1 < mean_magnitude < 2.5  # 扩大运动幅度范围
        and len(good_prev) > 20  # 降低所需特征点数量
    )

    return {
        "vertical_flow": float(vertical_flow),
        "horizontal_flow": float(horizontal_flow),
        "mean_magnitude": float(mean_magnitude),
        "max_magnitude": float(max_magnitude),
        "motion_consistency": float(dominant_motion),
        "motion_uniformity": float(motion_uniformity),
        "is_gradual": bool(is_gradual),
    }


def detect_slide_changes(
    video_path,
    threshold=DEFAULT_THRESHOLD,
    min_interval=DEFAULT_MIN_INTERVAL,
    debug_timestamps=None,
):
    """检测视频中的幻灯片变化，包括渐变检测"""
    start_time = datetime.now()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps
    video_duration = timedelta(seconds=total_seconds)
    min_frames = int(fps * min_interval)
    frame_count = 0
    raw_timestamps = []  # 存储原始时间戳
    potential_changes = []  # 存储潜在的变化点

    print(f"开始处理视频: {video_path}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"视频信息: {total_frames} 帧, {fps} FPS, 时长: {str(video_duration).split('.')[0]}"
    )
    print(f"检测参数: 阈值={threshold}, 最小间隔={min_interval}秒")
    if debug_timestamps:
        print(f"调试时间点: {', '.join(debug_timestamps)}")

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        return []

    # 分析第一帧
    (
        prev_hist,
        prev_density,
        prev_gray,
        _,
        prev_edge_density,
        prev_contrast,
        prev_vertical_diff,
        prev_horizontal_diff,
        prev_density_variance,
        prev_brightness,
        prev_contrast_brightness,
    ) = analyze_frame(prev_frame)
    frames_since_last_change = min_frames

    # 使用较小的跳帧值，在速度和准确性之间取平衡
    skip_frames = 0  # 取消跳帧以避免漏检快速变化

    # 动画检测相关变量
    animation_window = []  # 存储动画窗口内的变化
    animation_duration = int(fps * 1.0)  # 1秒的动画窗口
    in_animation = False
    animation_start_frame = 0

    # 渐变检测窗口
    fade_window = []
    fade_window_size = 5  # 用于检测淡入淡出的窗口大小

    while True:
        # 跳过部分帧以提高性能
        if skip_frames > 0:
            for _ in range(skip_frames):
                ret = cap.grab()
                if not ret:
                    break
                frame_count += 1

        ret, frame = cap.retrieve() if skip_frames > 0 else cap.read()
        if not ret:
            break

        frame_count += 1

        # 分析当前帧
        (
            curr_hist,
            curr_density,
            curr_gray,
            _,
            curr_edge_density,
            curr_contrast,
            curr_vertical_diff,
            curr_horizontal_diff,
            curr_density_variance,
            curr_brightness,
            curr_contrast_brightness,
        ) = analyze_frame(frame)

        if frames_since_last_change >= min_frames:
            # 基本变化检测
            correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_INTERSECT)
            correlation = correlation / np.sum(prev_hist)

            density_change = abs(curr_density - prev_density)
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            mean_diff = np.mean(frame_diff) / 255.0

            # 计算亮度和对比度变化
            brightness_change = abs(curr_brightness - prev_brightness) / 255.0
            contrast_change = abs(curr_contrast - prev_contrast) / 255.0

            # 更新淡入淡出检测窗口
            fade_window.append(
                {
                    "frame": frame_count,
                    "brightness": curr_brightness,
                    "contrast": curr_contrast,
                    "correlation": correlation,
                }
            )

            if len(fade_window) > fade_window_size:
                fade_window.pop(0)

            # 检测淡入淡出效果
            is_fade_transition = False
            if len(fade_window) == fade_window_size:
                brightness_trend = [frame["brightness"] for frame in fade_window]
                contrast_trend = [frame["contrast"] for frame in fade_window]
                correlation_trend = [frame["correlation"] for frame in fade_window]

                # 检查亮度变化趋势
                brightness_diff = np.diff(brightness_trend)
                contrast_diff = np.diff(contrast_trend)

                is_fade_transition = (
                    (
                        np.all(brightness_diff > 0) or np.all(brightness_diff < 0)
                    )  # 亮度持续变化
                    and abs(brightness_trend[-1] - brightness_trend[0])
                    > 10  # 亮度变化显著
                    and abs(contrast_trend[-1] - contrast_trend[0])
                    > 5  # 对比度变化显著
                    and min(correlation_trend) < 0.95  # 相关性有明显变化
                )

            # 计算方向性变化
            vertical_change = abs(curr_vertical_diff - prev_vertical_diff)
            horizontal_change = abs(curr_horizontal_diff - prev_horizontal_diff)
            directional_change = max(vertical_change, horizontal_change)

            # 计算光流特征
            flow_features = analyze_optical_flow(prev_gray, curr_gray)

            # 检测基于光流的翻页特征
            is_flow_change = (
                flow_features["motion_consistency"] > 0.3
                and flow_features["max_magnitude"] > 2.0  # 要求有显著的最大运动
                and flow_features["motion_uniformity"] < 0.8  # 运动应该有一定的不均匀性
                and (
                    (  # 垂直翻页
                        flow_features["vertical_flow"] > 0.8  # 提高垂直流阈值
                        and flow_features["horizontal_flow"] < 0.4  # 限制水平流
                        and flow_features["mean_magnitude"] > 0.5  # 要求足够的平均运动
                    )
                    or (  # 整体性强的运动
                        flow_features["mean_magnitude"] > 1.0  # 提高平均幅度要求
                        and flow_features["motion_consistency"] > 0.4  # 提高一致性要求
                        and flow_features["max_magnitude"] > 3.0  # 要求有显著的峰值运动
                    )
                )
            )

            # 优化的突变检测条件
            is_sudden_change = (
                correlation < 0.92  # 降低相关性阈值
                or (
                    mean_diff > threshold * 0.10  # 进一步降低整体变化阈值
                    and (
                        density_change > threshold * 0.06  # 降低密度变化阈值
                        or directional_change > threshold * 0.12  # 降低方向变化阈值
                        or (
                            curr_edge_density > 0.06
                            and abs(curr_edge_density - prev_edge_density)
                            > threshold * 0.10
                        )  # 降低边缘变化要求
                    )
                )
                or (  # 优化高相似度突变检测
                    correlation > 0.98  # 高相似度
                    and (
                        (  # 条件1：微小但显著的整体变化
                            mean_diff > threshold * 0.03  # 降低阈值
                            and abs(curr_brightness - prev_brightness)
                            > 3  # 降低亮度变化要求
                            and abs(curr_contrast - prev_contrast)
                            > 2  # 降低对比度变化要求
                        )
                        or (  # 条件2：边缘或文本的细微变化
                            abs(curr_edge_density - prev_edge_density)
                            > threshold * 0.05
                            and abs(curr_density - prev_density) > threshold * 0.03
                        )
                        or (  # 条件3：方向性变化
                            (vertical_change > 0.5 or horizontal_change > 0.5)
                            and directional_change > threshold * 0.08
                        )
                    )
                )
            )

            # 检测显著的垂直变化
            is_vertical_change = False
            if not is_sudden_change and frames_since_last_change >= min_frames:
                is_vertical_change = (
                    correlation > 0.95  # 降低相关性要求
                    and vertical_change > 1.2  # 降低垂直变化阈值
                    and horizontal_change < 0.5  # 放宽水平变化限制
                    and mean_diff < 0.04  # 放宽整体变化限制
                    and density_change < 0.003  # 放宽密度变化限制
                    and frames_since_last_change >= min_frames  # 减少所需间隔
                )

            # 检测显著的内容变化
            is_content_change = False
            if (
                not is_sudden_change
                and not is_vertical_change
                and frames_since_last_change >= min_frames
            ):
                is_content_change = (
                    correlation > 0.95  # 降低相关性要求
                    and vertical_change > 0.8  # 降低垂直变化要求
                    and horizontal_change > 0.8  # 降低水平变化要求
                    and abs(vertical_change - horizontal_change)
                    < 0.4  # 放宽变化差异限制
                    and curr_edge_density > 0.02  # 降低边缘密度要求
                    and frames_since_last_change >= min_frames  # 减少所需间隔
                )

            # 检测条件
            is_change = (
                is_sudden_change
                or is_vertical_change
                or is_content_change
                or is_flow_change
                or is_fade_transition
            )

            if is_change:
                current_time = timedelta(seconds=frame_count / fps)

                # 确定变化类型
                if is_fade_transition:
                    change_type = "fade"
                elif is_flow_change:
                    change_type = "flow"
                elif is_sudden_change:
                    change_type = "sudden"
                elif is_vertical_change:
                    change_type = "vertical"
                else:
                    change_type = "content"

                raw_timestamps.append(
                    {
                        "time": current_time,
                        "frame": frame_count,
                        "type": change_type,
                        "flow_features": (
                            flow_features if change_type == "flow" else None
                        ),
                    }
                )

                frames_since_last_change = 0
                fade_window.clear()  # 清空淡入淡出检测窗口

                print(f"\n检测到{change_type}变化: {format_timedelta(current_time)}")
                if change_type == "fade":
                    print("- 检测到淡入淡出效果")
                    print(f"- 亮度变化: {brightness_change:.3f}")
                    print(f"- 对比度变化: {contrast_change:.3f}")
                elif change_type == "flow":
                    print(
                        f"- 光流特征: 垂直={flow_features['vertical_flow']:.3f}, "
                        f"水平={flow_features['horizontal_flow']:.3f}, "
                        f"一致性={flow_features['motion_consistency']:.3f}"
                    )
                else:
                    print(f"- 直方图相关性: {correlation:.3f}")
                    if change_type == "vertical":
                        print(f"- 垂直变化值: {vertical_change:.3f}")
                    elif change_type == "content":
                        print(f"- 方向变化值: {directional_change:.3f}")
                    else:
                        print(f"- 区域变化值: {mean_diff:.3f}")

        # 显示处理进度
        if frame_count % int(fps) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\r处理进度: {progress:.1f}% ({frame_count}/{total_frames})", end="")

        # 更新上一帧的信息
        prev_hist = curr_hist
        prev_density = curr_density
        prev_gray = curr_gray
        prev_edge_density = curr_edge_density
        prev_contrast = curr_contrast
        prev_vertical_diff = curr_vertical_diff
        prev_horizontal_diff = curr_horizontal_diff
        prev_density_variance = curr_density_variance
        prev_brightness = curr_brightness
        prev_contrast_brightness = curr_contrast_brightness
        if not in_animation:
            frames_since_last_change += 1 + skip_frames

    cap.release()
    end_time = datetime.now()
    print(f"\n\n处理完成!")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {str(end_time - start_time).split('.')[0]}")
    print(f"检测到 {len(raw_timestamps)} 个变化点")
    print("\n变化点列表:")
    for i, ts in enumerate(raw_timestamps, 1):
        print(f"{i}. {format_timedelta(ts['time'])}")

    return raw_timestamps


def export_pr_markers(timestamps, video_path, output_file=None):
    """导出为Premiere Pro可用的标记文件"""
    if output_file is None:
        filename = os.path.basename(video_path)
        output_file = os.path.splitext(filename)[0] + "_markers.csv"
        output_file = os.path.join(os.path.dirname(video_path), output_file)

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=",")
        # 写入标题行
        writer.writerow(["标记名称", "描述", "入点", "出点", "持续时间", "标记类型"])

        # 写入每个标记
        for ts in timestamps:
            time_str = format_timedelta(ts["time"])
            writer.writerow(
                [
                    "幻灯片变化",
                    "检测到幻灯片切换",
                    time_str,
                    time_str,
                    "00:00:00:00",
                    "注释",
                ]
            )

    return output_file


import argparse
from src.detector.slide_detector import (
    detect_slide_changes,
    DEFAULT_THRESHOLD,
    DEFAULT_MIN_INTERVAL,
)
from src.exporters.csv_exporter import export_pr_markers
from src.exporters.xml_exporter import export_to_fcpxml


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
