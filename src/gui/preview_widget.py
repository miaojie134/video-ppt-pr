from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # 视频预览标签
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.preview_label)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)

    def start_preview(self, video_path):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            self.timer.start(33)  # 约30fps

    def stop_preview(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_preview(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # 转换图像格式并显示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            scaled_pixmap = QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
