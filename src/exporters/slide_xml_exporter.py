import cv2
import os
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any
from .xml_exporter import export_to_fcpxml as export_base_xml
from ..matcher.slide_matcher import SlideMatcher


def export_to_slide_xml(
    timestamps: List[Dict[str, Any]],
    video_path: str,
    slides_dir: str,
    output_file: str = None,
) -> str:
    """
    生成包含幻灯片替换的XML文件

    Args:
        timestamps: 检测到的时间点列表
        video_path: 视频文件路径
        slides_dir: 幻灯片图片目录
        output_file: 输出文件路径

    Returns:
        生成的XML文件路径
    """
    print("开始处理幻灯片匹配...")

    # 1. 首先进行幻灯片匹配
    matcher = SlideMatcher(video_path, slides_dir)
    matched_timestamps = matcher.match_timestamps(timestamps)

    # 2. 生成基础XML文件
    if output_file is None:
        base_name = os.path.splitext(video_path)[0]
        output_file = f"{base_name}_with_slides.xml"

    # 3. 读取基础XML文件内容
    base_xml = export_base_xml(matched_timestamps, video_path, output_file)
    if not base_xml:
        return None

    try:
        # 4. 解析XML
        tree = ET.parse(base_xml)
        root = tree.getroot()

        # 5. 找到视频轨道
        sequence = root.find(".//sequence")
        if sequence is None:
            raise ValueError("无法找到sequence元素")

        media = sequence.find("media")
        if media is None:
            raise ValueError("无法找到media元素")

        video = media.find("video")
        if video is None:
            raise ValueError("无法找到video元素")

        # 6. 添加幻灯片轨道
        slides_track = ET.SubElement(
            video,
            "track",
            {
                "TL.SQTrackShy": "0",
                "TL.SQTrackExpandedHeight": "25",
                "TL.SQTrackExpanded": "0",
                "MZ.TrackTargeted": "1",
            },
        )

        # 7. 为每个匹配的时间点添加幻灯片
        for i, ts in enumerate(matched_timestamps):
            if "matched_slide" not in ts:
                continue

            # 创建幻灯片文件引用
            slide_file = ET.Element("file", {"id": f"slide-{i+1}"})
            slide_name = ET.SubElement(slide_file, "name")
            slide_name.text = ts["matched_slide"]
            slide_pathurl = ET.SubElement(slide_file, "pathurl")
            slide_pathurl.text = f"file://localhost{os.path.abspath(ts['slide_path'])}"

            # 添加幻灯片片段
            clipitem = ET.SubElement(
                slides_track, "clipitem", {"id": f"slide-clip-{i+1}"}
            )

            # 设置片段名称
            clip_name = ET.SubElement(clipitem, "name")
            clip_name.text = ts["matched_slide"]

            # 设置片段启用状态
            clip_enabled = ET.SubElement(clipitem, "enabled")
            clip_enabled.text = "TRUE"

            # 计算时间点
            start_frame = int(ts["time"].total_seconds() * matcher.fps)
            if i < len(matched_timestamps) - 1:
                end_frame = int(
                    matched_timestamps[i + 1]["time"].total_seconds() * matcher.fps
                )
            else:
                end_frame = int(matcher.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 设置时间信息
            clip_start = ET.SubElement(clipitem, "start")
            clip_start.text = str(start_frame)
            clip_end = ET.SubElement(clipitem, "end")
            clip_end.text = str(end_frame)
            clip_in = ET.SubElement(clipitem, "in")
            clip_in.text = "0"
            clip_out = ET.SubElement(clipitem, "out")
            clip_out.text = str(end_frame - start_frame)

            # 添加Premier Pro特定时间码
            clip_ppro_in = ET.SubElement(clipitem, "pproTicksIn")
            clip_ppro_in.text = str(start_frame * 10160000)
            clip_ppro_out = ET.SubElement(clipitem, "pproTicksOut")
            clip_ppro_out.text = str(end_frame * 10160000)

            # 添加文件引用
            clipitem.append(slide_file)

        # 8. 保存修改后的XML
        xml_str = ET.tostring(root, encoding="unicode")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n')
            f.write(pretty_xml[pretty_xml.find("<xmeml") :])

        print(f"已生成带幻灯片的XML文件: {output_file}")
        return output_file

    except Exception as e:
        print(f"生成幻灯片XML时出错: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None
