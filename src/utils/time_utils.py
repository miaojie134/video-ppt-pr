from datetime import timedelta


def format_timedelta(td):
    """将timedelta转换为HH:MM:SS:FF格式（帧数）"""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 假设25帧/秒，将毫秒转换为帧号(0-24)
    frames = int((td.microseconds / 1000000) * 25)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
