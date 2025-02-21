import cv2
import numpy as np
from datetime import datetime, timedelta
from ..analyzer.frame_analyzer import analyze_frame
from ..analyzer.optical_flow import analyze_optical_flow
from ..utils.time_utils import format_timedelta

# 全局默认参数
DEFAULT_THRESHOLD = 0.25
DEFAULT_MIN_INTERVAL = 1.0

# 状态定义
STABLE = "STABLE"  # 稳定状态
CHANGING = "CHANGING"  # 变化状态
CONFIRMING = "CONFIRMING"  # 确认状态


def detect_slide_changes(
    video_path,
    threshold=DEFAULT_THRESHOLD,
    min_interval=DEFAULT_MIN_INTERVAL,
    debug_timestamps=None,
    progress_callback=None,
    status_callback=None,
    stop_check=None,
):
    """
    检测视频中的幻灯片变化，包括渐变检测

    参数:
        video_path: 视频文件路径
        threshold: 检测阈值
        min_interval: 最小检测间隔
        debug_timestamps: 需要调试输出的时间点列表，格式为["HH:MM:SS:FF", ...]
        progress_callback: 进度回调函数，接收0-100的进度值
        status_callback: 状态回调函数，接收状态信息字符串
        stop_check: 停止检查函数，返回True时中断处理
    """
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
    raw_timestamps = []
    fade_window = []
    fade_window_size = 5

    # 状态机相关变量
    current_state = STABLE
    state_start_frame = 0
    change_type = None
    stable_frames = 0
    change_features = {
        "start_hist": None,
        "start_density": None,
        "max_flow": 0,
        "total_movement": 0,
        "movement_direction": [],
        "correlation_trend": [],
        "last_significant_motion": 0,
    }

    # 动画检测相关变量
    animation_window = []
    animation_window_size = 50
    in_animation = False
    animation_start_frame = 0
    animation_features = {
        "start_hist": None,
        "start_density": None,
        "max_flow": 0,
        "total_movement": 0,
        "movement_direction": [],
        "correlation_trend": [],
        "last_significant_motion": 0,
        "start_time": None,
        "consistent_direction_count": 0,
        "primary_direction": None,
        "motion_start_frame": 0,
        "last_motion_magnitude": 0,
    }

    # 将debug_timestamps转换为帧号
    debug_frames = set()
    if debug_timestamps:
        for ts in debug_timestamps:
            if "-" in ts:  # 处理时间区间
                start_ts, end_ts = ts.split("-")
                # 解析开始时间
                h1, m1, s1, f1 = map(int, start_ts.split(":"))
                total_seconds_start = h1 * 3600 + m1 * 60 + s1
                frame_number_start = int(total_seconds_start * fps + f1)
                # 解析结束时间
                h2, m2, s2, f2 = map(int, end_ts.split(":"))
                total_seconds_end = h2 * 3600 + m2 * 60 + s2
                frame_number_end = int(total_seconds_end * fps + f2)
                # 添加区间内的所有帧
                debug_frames.update(range(frame_number_start, frame_number_end + 1))
            else:  # 处理单个时间点
                h, m, s, f = map(int, ts.split(":"))
                total_seconds = h * 3600 + m * 60 + s
                frame_number = int(total_seconds * fps + f)
                debug_frames.add(frame_number)

    print(f"开始处理视频: {video_path}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"视频信息: {total_frames} 帧, {fps} FPS, 时长: {str(video_duration).split('.')[0]}"
    )
    print(f"检测参数: 阈值={threshold}, 最小间隔={min_interval}秒")

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

    class SlideChangeDetector:
        def __init__(self, history_size=50):
            self.history_size = history_size
            self.feature_history = []
            self.change_history = []
            self.last_change_frame = 0
            self.min_change_interval = 25  # 1秒
            self.adaptive_thresholds = {
                "correlation": 0.98,
                "density": 0.05,
                "motion": 0.8,
            }

        def update_history(self, features):
            """更新特征历史"""
            self.feature_history.append(features)
            if len(self.feature_history) > self.history_size:
                self.feature_history.pop(0)

        def analyze_feature_distribution(self):
            """分析特征分布，更新自适应阈值"""
            if len(self.feature_history) < 10:
                return

            # 计算特征的统计信息
            correlations = [f["correlation"] for f in self.feature_history]
            densities = [f["density_change"] for f in self.feature_history]
            motions = [f["motion_magnitude"] for f in self.feature_history]

            # 使用百分位数设置阈值
            self.adaptive_thresholds["correlation"] = np.percentile(correlations, 10)
            self.adaptive_thresholds["density"] = np.percentile(densities, 90)
            self.adaptive_thresholds["motion"] = np.percentile(motions, 90)

        def is_significant_change(self, curr_features, frame_count):
            """判断是否为显著变化"""
            if frame_count - self.last_change_frame < self.min_change_interval:
                return False, None

            # 更新特征历史
            self.update_history(curr_features)
            self.analyze_feature_distribution()

            # 提取当前特征
            correlation = curr_features["correlation"]
            density_change = curr_features["density_change"]
            motion_magnitude = curr_features["motion_magnitude"]
            edge_change = curr_features["edge_change"]
            text_structure_change = curr_features["text_structure_change"]
            flow_features = curr_features["flow_features"]

            # 计算基础变化分数
            base_score = 0

            # 1. 相关性变化评分
            if correlation < 0.90:  # 显著变化
                base_score += 2
            elif correlation < 0.95:  # 中等变化
                base_score += 1

            # 2. 密度变化评分
            if density_change > 0.05:  # 显著变化
                base_score += 2
            elif density_change > 0.02:  # 中等变化
                base_score += 1

            # 3. 文本结构变化评分
            if text_structure_change > 0.1:  # 显著变化
                base_score += 2
            elif text_structure_change > 0.05:  # 中等变化
                base_score += 1

            # 4. 边缘变化评分
            if edge_change > 0.02:
                base_score += 1

            # 5. 运动特征评分
            if flow_features["is_gradual"]:
                if motion_magnitude > 1.0 and flow_features["motion_coherence"] > 0.7:
                    base_score += 2
                elif motion_magnitude > 0.5 and flow_features["motion_coherence"] > 0.5:
                    base_score += 1

            # 检查是否是讲师动作
            is_lecturer_motion = flow_features["is_lecturer_motion"]

            # 确定变化类型
            change_type = None
            final_score = 0

            # 根据特征组合确定变化类型和最终分数
            if not is_lecturer_motion:  # 如果不是讲师动作
                if correlation < 0.95 and (
                    density_change > 0.02 or text_structure_change > 0.05
                ):
                    change_type = "sudden"
                    final_score = base_score
                elif flow_features["is_gradual"] and motion_magnitude > 0.5:
                    change_type = "gradual"
                    final_score = base_score
                elif text_structure_change > 0.05 and density_change > 0.01:
                    change_type = "static"
                    final_score = base_score
                # 增加对大幅度运动的特殊处理
                elif motion_magnitude > 20.0:  # 运动幅度特别大
                    if (
                        density_change > 0.01  # 降低密度变化要求
                        or text_structure_change > 0.02  # 降低文本结构变化要求
                        or edge_change > 0.01  # 降低边缘变化要求
                    ):
                        change_type = "motion"
                        final_score = 3  # 给予较高分数
            else:
                # 如果是讲师动作，需要更高的阈值
                if base_score >= 4:  # 要求更高的分数
                    if correlation < 0.90 and density_change > 0.05:
                        change_type = "sudden"
                        final_score = base_score - 1  # 降低分数
                    elif flow_features["is_gradual"] and motion_magnitude > 1.5:
                        change_type = "gradual"
                        final_score = base_score - 1

            # 时间间隔验证
            if change_type and len(self.change_history) > 0:
                last_change = self.change_history[-1]
                time_since_last = frame_count - last_change["frame"]

                # 如果距离上次变化太近，需要更严格的判断
                if time_since_last < self.min_change_interval * 2:
                    if final_score < 3:  # 要求更高的分数
                        return False, None

            # 最终判断
            is_change = final_score >= 2  # 降低最低要求

            if is_change:
                self.last_change_frame = frame_count
                self.change_history.append(
                    {
                        "frame": frame_count,
                        "type": change_type,
                        "score": final_score,
                        "features": curr_features,
                    }
                )

            return is_change, change_type

    # 初始化检测器
    detector = SlideChangeDetector()

    while True:
        # 检查是否需要停止
        if stop_check and stop_check():
            print("\n检测已中断")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 分析当前帧
        (
            curr_hist,
            curr_density,
            curr_gray,
            curr_text_mask,
            curr_edge_density,
            curr_contrast,
            curr_vertical_diff,
            curr_horizontal_diff,
            curr_density_variance,
            curr_brightness,
            curr_contrast_brightness,
        ) = analyze_frame(frame)

        if frames_since_last_change >= min_frames:
            # 基本特征提取
            correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_INTERSECT)
            correlation = correlation / np.sum(prev_hist)

            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            mean_diff = np.mean(frame_diff) / 255.0
            density_change = abs(curr_density - prev_density)
            edge_change = abs(curr_edge_density - prev_edge_density)

            # 计算光流特征
            flow_features = analyze_optical_flow(prev_gray, curr_gray)

            # 分析文本结构
            curr_structure = analyze_text_structure(curr_text_mask)
            prev_structure = analyze_text_structure(
                prev_text_mask
                if "prev_text_mask" in locals()
                else np.zeros_like(curr_text_mask)
            )

            # 计算文本结构变化
            text_structure_change = 0
            if curr_structure and prev_structure:
                area_change = abs(
                    curr_structure["total_area"] - prev_structure["total_area"]
                ) / max(curr_structure["total_area"], prev_structure["total_area"])
                count_change = abs(
                    curr_structure["num_regions"] - prev_structure["num_regions"]
                ) / max(curr_structure["num_regions"], prev_structure["num_regions"])
                text_structure_change = (area_change + count_change) / 2

            # 构建特征字典
            current_features = {
                "correlation": correlation,
                "density_change": density_change,
                "motion_magnitude": flow_features["mean_magnitude"],
                "edge_change": edge_change,
                "text_structure_change": text_structure_change,
                "flow_features": flow_features,
            }

            # 使用检测器判断变化
            is_change, change_type = detector.is_significant_change(
                current_features, frame_count
            )

            # 调试信息输出
            if debug_timestamps and frame_count in debug_frames:
                current_time = timedelta(seconds=frame_count / fps)
                print(f"\n调试信息 [{format_timedelta(current_time)}]:")
                print(f"帧号: {frame_count}")
                print("基础特征:")
                print(f"- 相关性: {correlation:.4f}")
                print(f"- 密度变化: {density_change:.4f}")
                print(f"- 边缘变化: {edge_change:.4f}")
                print(f"- 文本结构变化: {text_structure_change:.4f}")
                print(f"- 运动幅度: {flow_features['mean_magnitude']:.4f}")
                print("光流特征:")
                print(f"- 是否为渐变: {flow_features['is_gradual']}")
                print(f"- 运动连贯性: {flow_features['motion_coherence']:.4f}")
                print(f"- 是否为讲师动作: {flow_features['is_lecturer_motion']}")
                print(f"- 讲师运动比例: {flow_features['lecturer_movement_ratio']:.4f}")
                if curr_structure:
                    print("文本结构:")
                    print(f"- 文本区域数量: {curr_structure['num_regions']}")
                    print(f"- 平均区域高度: {curr_structure['avg_height']:.2f}")
                    print(f"- 平均区域宽度: {curr_structure['avg_width']:.2f}")
                    print(f"- 总文本面积: {curr_structure['total_area']:.2f}")
                print(f"检测结果: {'检测到变化' if is_change else '未检测到变化'}")
                if is_change:
                    print(f"变化类型: {change_type}")
                print("-" * 50)

            if is_change:
                # 记录变化点
                current_time = timedelta(seconds=frame_count / fps)
                change_message = f"\n检测到{change_type}变化: {format_timedelta(current_time)}"
                
                # 发送变化检测信息
                if status_callback:
                    status_callback(change_message)
                    
                    # 发送详细信息
                    details = [
                        f"- 相关性: {correlation:.3f}",
                        f"- 密度变化: {density_change:.3f}",
                        f"- 边缘变化: {edge_change:.3f}",
                        f"- 文本结构变化: {text_structure_change:.3f}"
                    ]
                    
                    if flow_features["mean_magnitude"] > 0:
                        details.extend([
                            f"- 运动特征: {flow_features['mean_magnitude']:.3f}",
                            f"- 运动连贯性: {flow_features['motion_coherence']:.3f}",
                            f"- 讲师动作: {flow_features['is_lecturer_motion']}"
                        ])
                    
                    for detail in details:
                        status_callback(detail)

                raw_timestamps.append({
                    "time": current_time,
                    "frame": frame_count,
                    "type": change_type,
                    "features": current_features
                })

                frames_since_last_change = 0

        # 更新进度
        if frame_count % int(fps) == 0:
            progress = (frame_count / total_frames) * 100
            if progress_callback:
                progress_callback(int(progress))
            if status_callback:
                status_callback(
                    f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})"
                )

        # 更新上一帧的信息
        prev_hist = curr_hist
        prev_density = curr_density
        prev_gray = curr_gray
        prev_text_mask = curr_text_mask
        prev_edge_density = curr_edge_density
        prev_contrast = curr_contrast
        prev_vertical_diff = curr_vertical_diff
        prev_horizontal_diff = curr_horizontal_diff
        prev_density_variance = curr_density_variance
        prev_brightness = curr_brightness
        prev_contrast_brightness = curr_contrast_brightness
        frames_since_last_change += 1

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


def analyze_layout_change(
    curr_frame_data, prev_frame_data, curr_text_mask, prev_text_mask
):
    """分析幻灯片布局变化"""
    # 将图像分为上中下三个区域
    height, width = curr_text_mask.shape
    top_region = slice(0, height // 4)
    middle_region = slice(height // 4, 3 * height // 4)
    bottom_region = slice(3 * height // 4, height)

    # 分析各区域的文本密度变化
    def get_region_density(mask, region):
        region_mask = mask[region, :]
        return np.sum(region_mask > 0) / region_mask.size if region_mask.size > 0 else 0

    # 计算各区域的变化
    top_change = abs(
        get_region_density(curr_text_mask, top_region)
        - get_region_density(prev_text_mask, top_region)
    )
    middle_change = abs(
        get_region_density(curr_text_mask, middle_region)
        - get_region_density(prev_text_mask, middle_region)
    )
    bottom_change = abs(
        get_region_density(curr_text_mask, bottom_region)
        - get_region_density(prev_text_mask, bottom_region)
    )

    # 分析标题区域
    title_change = top_change > 0.1  # 标题变化阈值

    # 分析内容区域
    content_change = middle_change > 0.05  # 内容变化阈值

    # 计算整体布局变化分数
    layout_score = {
        "title_change": title_change,
        "content_change": content_change,
        "top_change": top_change,
        "middle_change": middle_change,
        "bottom_change": bottom_change,
        "is_significant": (title_change or (content_change and middle_change > 0.08)),
    }

    return layout_score


def analyze_text_structure(text_mask):
    """分析文本结构特征"""
    # 使用连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_mask.astype(np.uint8)
    )

    if num_labels < 2:  # 如果没有文本区域
        return None

    # 过滤掉太小的区域
    valid_regions = stats[1:][stats[1:, cv2.CC_STAT_AREA] > 100]

    if len(valid_regions) == 0:
        return None

    # 计算文本块的分布特征
    heights = valid_regions[:, cv2.CC_STAT_HEIGHT]
    widths = valid_regions[:, cv2.CC_STAT_WIDTH]
    areas = valid_regions[:, cv2.CC_STAT_AREA]

    # 计算特征
    avg_height = np.mean(heights)
    avg_width = np.mean(widths)
    avg_area = np.mean(areas)

    return {
        "num_regions": len(valid_regions),
        "avg_height": avg_height,
        "avg_width": avg_width,
        "avg_area": avg_area,
        "total_area": np.sum(areas),
    }


def verify_change(
    curr_frame_data, prev_frame_data, curr_text_mask, prev_text_mask, flow_features
):
    """多阶段验证机制"""
    # 1. 分析布局变化
    layout_change = analyze_layout_change(
        curr_frame_data, prev_frame_data, curr_text_mask, prev_text_mask
    )

    # 2. 分析文本结构
    curr_structure = analyze_text_structure(curr_text_mask)
    prev_structure = analyze_text_structure(prev_text_mask)

    if curr_structure is None or prev_structure is None:
        return False, None

    # 3. 计算结构变化
    structure_change = abs(
        curr_structure["total_area"] - prev_structure["total_area"]
    ) / max(curr_structure["total_area"], prev_structure["total_area"])
    region_count_change = abs(
        curr_structure["num_regions"] - prev_structure["num_regions"]
    ) / max(curr_structure["num_regions"], prev_structure["num_regions"])

    # 4. 综合判断
    is_title_change = layout_change["title_change"]
    is_content_change = layout_change["content_change"]
    is_structure_change = structure_change > 0.15 or region_count_change > 0.2

    # 确定变化类型
    if is_title_change and is_content_change:
        change_type = "full"  # 整页变化
    elif is_title_change:
        change_type = "title"  # 标题变化
    elif is_content_change and is_structure_change:
        change_type = "content"  # 内容变化
    else:
        return False, None

    # 5. 最终验证
    # 检查是否有足够的变化特征
    min_change_score = 2
    change_score = 0

    # 累计变化分数
    if layout_change["is_significant"]:
        change_score += 1
    if structure_change > 0.15:
        change_score += 1
    if region_count_change > 0.2:
        change_score += 1
    if layout_change["top_change"] > 0.1:
        change_score += 1
    if layout_change["middle_change"] > 0.08:
        change_score += 1

    return change_score >= min_change_score, change_type
