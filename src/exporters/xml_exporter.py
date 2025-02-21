import cv2
import os
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ..utils.time_utils import format_timedelta


def export_to_fcpxml(timestamps, video_path, output_file=None):
    """生成Final Cut Pro XML格式文件（可被Premiere Pro直接导入）"""
    print("开始生成XML文件...")

    if output_file is None:
        output_file = os.path.splitext(video_path)[0] + ".xml"
    print(f"输出文件路径: {output_file}")

    # 获取视频信息
    print("正在读取视频信息...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    cap.release()

    try:
        print("正在创建XML结构...")
        root = ET.Element("xmeml", {"version": "4"})

        # 添加序列
        sequence = ET.SubElement(
            root,
            "sequence",
            {
                "id": "sequence-1",
                "explodedTracks": "true",
                "TL.SQAudioVisibleBase": "0",
                "TL.SQVideoVisibleBase": "0",
                "TL.SQVisibleBaseTime": "0",
                "TL.SQAVDividerPosition": "0.5",
                "TL.SQHideShyTracks": "0",
                "Monitor.ProgramZoomOut": str(total_frames * 10160000),
                "Monitor.ProgramZoomIn": "0",
            },
        )

        # 添加序列UUID
        seq_uuid = ET.SubElement(sequence, "uuid")
        seq_uuid.text = str(uuid.uuid4())

        # 添加序列持续时间
        duration = ET.SubElement(sequence, "duration")
        duration.text = str(total_frames)

        # 添加序列帧率
        rate = ET.SubElement(sequence, "rate")
        timebase = ET.SubElement(rate, "timebase")
        timebase.text = str(int(round(fps)))
        ntsc = ET.SubElement(rate, "ntsc")
        ntsc.text = "FALSE"

        name = ET.SubElement(sequence, "name")
        name.text = os.path.splitext(os.path.basename(video_path))[0]

        # 添加媒体
        media = ET.SubElement(sequence, "media")

        # 添加视频部分
        video = ET.SubElement(media, "video")
        video_format = ET.SubElement(video, "format")
        video_samplecharacteristics = ET.SubElement(
            video_format, "samplecharacteristics"
        )

        # 视频帧率
        video_rate = ET.SubElement(video_samplecharacteristics, "rate")
        video_timebase = ET.SubElement(video_rate, "timebase")
        video_timebase.text = str(int(round(fps)))
        video_ntsc = ET.SubElement(video_rate, "ntsc")
        video_ntsc.text = "FALSE"

        # 视频尺寸
        video_width = ET.SubElement(video_samplecharacteristics, "width")
        video_width.text = str(width)
        video_height = ET.SubElement(video_samplecharacteristics, "height")
        video_height.text = str(height)
        video_anamorphic = ET.SubElement(video_samplecharacteristics, "anamorphic")
        video_anamorphic.text = "FALSE"
        video_pixelaspectratio = ET.SubElement(
            video_samplecharacteristics, "pixelaspectratio"
        )
        video_pixelaspectratio.text = "square"

        # 添加视频轨道
        video_track = ET.SubElement(
            video,
            "track",
            {
                "TL.SQTrackShy": "0",
                "TL.SQTrackExpandedHeight": "25",
                "TL.SQTrackExpanded": "0",
                "MZ.TrackTargeted": "1",
            },
        )

        # 创建文件引用
        file = ET.Element("file", {"id": "file-1"})
        file_name = ET.SubElement(file, "name")
        file_name.text = os.path.basename(video_path)
        pathurl = ET.SubElement(file, "pathurl")

        # 修改pathurl的生成逻辑
        # Windows 格式的路径处理
        if os.name == 'nt':
            # 转换为Windows格式的文件URL
            abs_path = os.path.abspath(video_path)
            # 确保使用正斜杠
            abs_path = abs_path.replace("\\", "/")
            # 添加额外的斜杠使其成为合法的file URL
            if not abs_path.startswith("/"):
                abs_path = "/" + abs_path
            pathurl.text = f"file://localhost{abs_path}"
        else:
            # macOS/Linux 格式的路径处理
            abs_path = os.path.abspath(video_path)
            pathurl.text = f"file://localhost{abs_path}"

        # 添加文件帧率
        file_rate = ET.SubElement(file, "rate")
        file_timebase = ET.SubElement(file_rate, "timebase")
        file_timebase.text = str(int(round(fps)))
        file_ntsc = ET.SubElement(file_rate, "ntsc")
        file_ntsc.text = "FALSE"

        # 添加文件持续时间
        file_duration = ET.SubElement(file, "duration")
        file_duration.text = str(total_frames)

        # 添加时间码
        timecode = ET.SubElement(file, "timecode")
        tc_rate = ET.SubElement(timecode, "rate")
        tc_timebase = ET.SubElement(tc_rate, "timebase")
        tc_timebase.text = str(int(round(fps)))
        tc_ntsc = ET.SubElement(tc_rate, "ntsc")
        tc_ntsc.text = "FALSE"
        tc_string = ET.SubElement(timecode, "string")
        tc_string.text = "00:00:00:00"
        tc_frame = ET.SubElement(timecode, "frame")
        tc_frame.text = "0"
        tc_displayformat = ET.SubElement(timecode, "displayformat")
        tc_displayformat.text = "NDF"

        # 添加媒体特征
        file_media = ET.SubElement(file, "media")

        # 视频信息
        file_video = ET.SubElement(file_media, "video")
        file_video_samplecharacteristics = ET.SubElement(
            file_video, "samplecharacteristics"
        )

        # 视频特征
        file_video_rate = ET.SubElement(file_video_samplecharacteristics, "rate")
        file_video_timebase = ET.SubElement(file_video_rate, "timebase")
        file_video_timebase.text = str(int(round(fps)))
        file_video_ntsc = ET.SubElement(file_video_rate, "ntsc")
        file_video_ntsc.text = "FALSE"

        file_video_width = ET.SubElement(file_video_samplecharacteristics, "width")
        file_video_width.text = str(width)
        file_video_height = ET.SubElement(file_video_samplecharacteristics, "height")
        file_video_height.text = str(height)
        file_video_anamorphic = ET.SubElement(
            file_video_samplecharacteristics, "anamorphic"
        )
        file_video_anamorphic.text = "FALSE"
        file_video_pixelaspectratio = ET.SubElement(
            file_video_samplecharacteristics, "pixelaspectratio"
        )
        file_video_pixelaspectratio.text = "square"

        # 音频信息
        file_audio = ET.SubElement(file_media, "audio")
        file_audio_samplecharacteristics = ET.SubElement(
            file_audio, "samplecharacteristics"
        )
        file_audio_depth = ET.SubElement(file_audio_samplecharacteristics, "depth")
        file_audio_depth.text = "16"
        file_audio_samplerate = ET.SubElement(
            file_audio_samplecharacteristics, "samplerate"
        )
        file_audio_samplerate.text = "44100"
        file_audio_channelcount = ET.SubElement(file_audio, "channelcount")
        file_audio_channelcount.text = "2"

        # 添加视频片段
        clipitem = ET.SubElement(video_track, "clipitem", {"id": "clipitem-1"})
        clip_name = ET.SubElement(clipitem, "name")
        clip_name.text = os.path.basename(video_path)
        clip_enabled = ET.SubElement(clipitem, "enabled")
        clip_enabled.text = "TRUE"
        clip_duration = ET.SubElement(clipitem, "duration")
        clip_duration.text = str(total_frames)

        # 添加片段帧率
        clip_rate = ET.SubElement(clipitem, "rate")
        clip_timebase = ET.SubElement(clip_rate, "timebase")
        clip_timebase.text = str(int(round(fps)))
        clip_ntsc = ET.SubElement(clip_rate, "ntsc")
        clip_ntsc.text = "FALSE"

        # 添加片段时间信息
        clip_start = ET.SubElement(clipitem, "start")
        clip_start.text = "0"
        clip_end = ET.SubElement(clipitem, "end")
        clip_end.text = str(total_frames)
        clip_in = ET.SubElement(clipitem, "in")
        clip_in.text = "0"
        clip_out = ET.SubElement(clipitem, "out")
        clip_out.text = str(total_frames)

        # 添加Premier Pro特定时间码
        clip_ppro_in = ET.SubElement(clipitem, "pproTicksIn")
        clip_ppro_in.text = "0"
        clip_ppro_out = ET.SubElement(clipitem, "pproTicksOut")
        clip_ppro_out.text = str(total_frames * 10160000)

        # 添加视频特性
        clip_alphatype = ET.SubElement(clipitem, "alphatype")
        clip_alphatype.text = "none"
        clip_pixelaspectratio = ET.SubElement(clipitem, "pixelaspectratio")
        clip_pixelaspectratio.text = "square"
        clip_anamorphic = ET.SubElement(clipitem, "anamorphic")
        clip_anamorphic.text = "FALSE"

        # 添加文件引用
        clipitem.append(file)

        # 添加音频部分
        audio = ET.SubElement(media, "audio")
        audio_numOutputChannels = ET.SubElement(audio, "numOutputChannels")
        audio_numOutputChannels.text = "2"

        # 添加音频格式
        audio_format = ET.SubElement(audio, "format")
        audio_samplecharacteristics = ET.SubElement(
            audio_format, "samplecharacteristics"
        )
        audio_depth = ET.SubElement(audio_samplecharacteristics, "depth")
        audio_depth.text = "16"
        audio_samplerate = ET.SubElement(audio_samplecharacteristics, "samplerate")
        audio_samplerate.text = "44100"

        # 添加音频轨道
        audio_track = ET.SubElement(
            audio,
            "track",
            {
                "TL.SQTrackAudioKeyframeStyle": "0",
                "TL.SQTrackShy": "0",
                "TL.SQTrackExpandedHeight": "25",
                "TL.SQTrackExpanded": "0",
                "MZ.TrackTargeted": "1",
                "PannerCurrentValue": "0.5",
                "PannerIsInverted": "true",
                "PannerStartKeyframe": "-91445760000000000,0.5,0,0,0,0,0,0",
                "PannerName": "平衡",
                "currentExplodedTrackIndex": "0",
                "totalExplodedTrackCount": "2",
                "premiereTrackType": "Stereo",
            },
        )

        # 添加音频轨道属性
        track_enabled = ET.SubElement(audio_track, "enabled")
        track_enabled.text = "TRUE"
        track_locked = ET.SubElement(audio_track, "locked")
        track_locked.text = "FALSE"
        track_outputchannelindex = ET.SubElement(audio_track, "outputchannelindex")
        track_outputchannelindex.text = "1"

        # 添加音频片段
        audio_clipitem = ET.SubElement(
            audio_track,
            "clipitem",
            {
                "id": "clipitem-2",
                "premiereChannelType": "stereo",
            },
        )

        # 添加音频片段属性
        audio_clip_name = ET.SubElement(audio_clipitem, "name")
        audio_clip_name.text = os.path.basename(video_path)
        audio_clip_enabled = ET.SubElement(audio_clipitem, "enabled")
        audio_clip_enabled.text = "TRUE"
        audio_clip_duration = ET.SubElement(audio_clipitem, "duration")
        audio_clip_duration.text = str(total_frames)

        # 添加音频片段时间信息
        audio_clip_start = ET.SubElement(audio_clipitem, "start")
        audio_clip_start.text = "0"
        audio_clip_end = ET.SubElement(audio_clipitem, "end")
        audio_clip_end.text = str(total_frames)
        audio_clip_in = ET.SubElement(audio_clipitem, "in")
        audio_clip_in.text = "0"
        audio_clip_out = ET.SubElement(audio_clipitem, "out")
        audio_clip_out.text = str(total_frames)

        # 添加Premier Pro特定时间码
        audio_clip_ppro_in = ET.SubElement(audio_clipitem, "pproTicksIn")
        audio_clip_ppro_in.text = "0"
        audio_clip_ppro_out = ET.SubElement(audio_clipitem, "pproTicksOut")
        audio_clip_ppro_out.text = str(total_frames * 10160000)

        # 添加音频文件引用
        audio_clipitem.append(file)

        # 添加链接关系
        clip_link = ET.SubElement(clipitem, "link")
        link_clipref = ET.SubElement(clip_link, "linkclipref")
        link_clipref.text = "clipitem-1"
        link_mediatype = ET.SubElement(clip_link, "mediatype")
        link_mediatype.text = "video"
        link_trackindex = ET.SubElement(clip_link, "trackindex")
        link_trackindex.text = "1"
        link_clipindex = ET.SubElement(clip_link, "clipindex")
        link_clipindex.text = "1"

        audio_link = ET.SubElement(clipitem, "link")
        audio_link_clipref = ET.SubElement(audio_link, "linkclipref")
        audio_link_clipref.text = "clipitem-2"
        audio_link_mediatype = ET.SubElement(audio_link, "mediatype")
        audio_link_mediatype.text = "audio"
        audio_link_trackindex = ET.SubElement(audio_link, "trackindex")
        audio_link_trackindex.text = "1"
        audio_link_clipindex = ET.SubElement(audio_link, "clipindex")
        audio_link_clipindex.text = "1"
        audio_link_groupindex = ET.SubElement(audio_link, "groupindex")
        audio_link_groupindex.text = "1"

        # 添加标记
        print("正在添加标记点...")
        markers = ET.SubElement(sequence, "markers")
        for i, timestamp in enumerate(timestamps, 1):
            marker = ET.SubElement(markers, "marker")
            marker_name = ET.SubElement(marker, "name")
            marker_name.text = f"Slide {i}"
            comment = ET.SubElement(marker, "comment")
            comment.text = f"Slide Change {i}"
            frame = int(timestamp["time"].total_seconds() * fps)
            in_point = ET.SubElement(marker, "in")
            in_point.text = str(frame)
            out_point = ET.SubElement(marker, "out")
            out_point.text = str(frame)

        print("正在格式化XML...")
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

        print("正在保存文件...")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n')
            f.write(xml_str[xml_str.find("<xmeml") :])

        print(f"已成功生成XML文件: {output_file}")
        print("\n重要提示：")
        print("1. 在Premiere Pro中导入XML文件前，请先导入视频文件")
        print("2. 确保视频文件路径没有变动")
        print(f"3. 当前视频文件完整路径: {os.path.abspath(video_path)}")
        print("4. 如果仍然提示视频脱机，请在PR中手动重新链接视频文件")
        return output_file

    except Exception as e:
        print(f"生成XML文件时发生错误: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None
