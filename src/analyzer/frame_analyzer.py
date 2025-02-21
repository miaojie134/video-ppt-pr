import cv2
import numpy as np


def analyze_frame(frame, content_mask=None):
    """分析帧的特征，支持动态内容区域"""
    height, width = frame.shape[:2]

    # 调整图像大小以提高性能
    frame = cv2.resize(frame, (960, 540))
    height, width = frame.shape[:2]

    # 如果没有提供内容掩码，使用默认的左侧2/3区域
    if content_mask is None:
        content_width = int(width * 0.67)
        content_mask = np.zeros((height, width), dtype=np.uint8)
        content_mask[:, :content_width] = 1

    # 应用内容掩码
    content_region = frame.copy()
    content_region[content_mask == 0] = 0

    # 转换为灰度图
    gray = cv2.cvtColor(content_region, cv2.COLOR_BGR2GRAY)

    # 1. 计算直方图
    hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 2. 计算亮度和对比度
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # 3. 改进的文本检测
    # 使用多尺度自适应阈值
    binary_results = []
    block_sizes = [11, 21, 31]
    for block_size in block_sizes:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            2,
        )
        binary_results.append(binary)

    # 合并多尺度结果
    text_mask = np.zeros_like(gray)
    for binary in binary_results:
        text_mask = cv2.bitwise_or(text_mask, binary)

    # 使用形态学操作优化文本区域
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 去除噪点
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel_small)
    # 连接相近文本
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_large)

    # 4. 改进的文本密度和分布分析
    rows, cols = 16, 16  # 增加网格密度
    cell_height = text_mask.shape[0] // rows
    cell_width = text_mask.shape[1] // cols
    region_densities = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            region = text_mask[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ]
            if content_mask[i * cell_height, j * cell_width] > 0:  # 只分析内容区域
                density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                region_densities[i, j] = density

    text_density = np.mean(region_densities[region_densities > 0])
    text_density_variance = np.var(region_densities[region_densities > 0])

    # 5. 改进的文本分布分析
    vertical_splits = 40  # 增加分割精度
    horizontal_splits = 40

    v_height = text_mask.shape[0] // vertical_splits
    h_width = text_mask.shape[1] // horizontal_splits

    vertical_profile = []
    horizontal_profile = []

    # 计算垂直分布
    for i in range(vertical_splits):
        section = text_mask[i * v_height : (i + 1) * v_height, :]
        mask_section = content_mask[i * v_height : (i + 1) * v_height, :]
        if np.sum(mask_section) > 0:  # 只分析内容区域
            density = np.sum(section) / np.sum(mask_section)
            vertical_profile.append(density)

    # 计算水平分布
    for i in range(horizontal_splits):
        section = text_mask[:, i * h_width : (i + 1) * h_width]
        mask_section = content_mask[:, i * h_width : (i + 1) * h_width]
        if np.sum(mask_section) > 0:  # 只分析内容区域
            density = np.sum(section) / np.sum(mask_section)
            horizontal_profile.append(density)

    # 使用中值滤波平滑分布曲线
    vertical_profile = np.array(vertical_profile)
    horizontal_profile = np.array(horizontal_profile)
    vertical_profile = cv2.medianBlur(vertical_profile.astype(np.float32), 3)
    horizontal_profile = cv2.medianBlur(horizontal_profile.astype(np.float32), 3)

    vertical_diff = np.mean(np.abs(np.diff(vertical_profile)))
    horizontal_diff = np.mean(np.abs(np.diff(horizontal_profile)))

    # 6. 改进的边缘特征分析
    edges = cv2.Canny(gray, 30, 150)
    edge_density = np.sum(edges > 0) / np.sum(content_mask)

    # 7. 改进的局部对比度分析
    kernel = np.ones((7, 7), np.float32) / 49
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
        brightness,
        contrast,
    )
