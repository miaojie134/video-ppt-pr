import csv
import os
from ..utils.time_utils import format_timedelta


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
