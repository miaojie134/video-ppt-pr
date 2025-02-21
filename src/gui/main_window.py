import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QDoubleSpinBox,
    QTextEdit,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ..detector.slide_detector import (
    detect_slide_changes,
    DEFAULT_THRESHOLD,
    DEFAULT_MIN_INTERVAL,
)
from datetime import datetime
from PyQt6.QtGui import QTextCursor
import os


class VideoProcessThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, video_path, threshold, min_interval):
        super().__init__()
        self.video_path = video_path
        self.threshold = threshold
        self.min_interval = min_interval
        self._is_running = True

    def run(self):
        try:

            def status_callback(message):
                self.status.emit(message)

            results = detect_slide_changes(
                self.video_path,
                self.threshold,
                self.min_interval,
                progress_callback=self.update_progress,
                status_callback=status_callback,
                stop_check=lambda: not self._is_running,
            )
            if self._is_running:
                self.finished.emit(results)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

    def stop(self):
        self._is_running = False

    def update_progress(self, value):
        self.progress.emit(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPT幻灯片变化检测工具")
        self.setMinimumSize(800, 600)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 文件选择区域
        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        self.file_label = QLabel("未选择文件")
        self.select_button = QPushButton("选择视频")
        self.select_button.clicked.connect(self.select_video)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_button)
        layout.addWidget(file_widget)

        # 状态显示
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 结果显示区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        # 控制按钮
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("停止检测")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        layout.addWidget(control_widget)

        self.video_path = None
        self.detection_results = None
        self.process_thread = None

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)"
        )
        if file_path:
            self.video_path = file_path
            self.file_label.setText(f"已选择: {file_path}")
            self.start_button.setEnabled(True)

    def start_detection(self):
        if not self.video_path:
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.result_text.clear()
        self.status_label.setText("正在处理...")

        # 创建处理线程，使用默认参数
        self.process_thread = VideoProcessThread(
            self.video_path,
            DEFAULT_THRESHOLD,  # 使用默认阈值
            DEFAULT_MIN_INTERVAL,  # 使用默认间隔
        )
        self.process_thread.progress.connect(self.update_progress)
        self.process_thread.status.connect(self.update_status)
        self.process_thread.finished.connect(self.detection_finished)
        self.process_thread.error.connect(self.handle_error)
        self.process_thread.start()

    def stop_detection(self):
        if self.process_thread and self.process_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认停止",
                "确定要停止检测吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.process_thread.stop()
                self.process_thread.wait()  # 等待线程结束
                self.status_label.setText("检测已中止")
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.progress_bar.setValue(0)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def detection_finished(self, results):
        self.detection_results = results
        self.display_results()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("检测完成")

        # 自动导出CSV和XML
        try:
            # 生成输出文件路径
            video_dir = os.path.dirname(self.video_path)
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]

            # 导出CSV
            csv_path = os.path.join(video_dir, f"{base_name}_markers.csv")
            from ..exporters.csv_exporter import export_pr_markers

            csv_file = export_pr_markers(
                self.detection_results, self.video_path, csv_path
            )

            # 导出XML
            xml_path = os.path.join(video_dir, f"{base_name}.xml")
            from ..exporters.xml_exporter import export_to_fcpxml

            xml_file = export_to_fcpxml(
                self.detection_results, self.video_path, xml_path
            )

            # 显示导出结果
            export_msg = "检测完成！已自动导出到:\n"
            if csv_file:
                export_msg += f"CSV标记文件: {csv_file}\n"
            if xml_file:
                export_msg += f"XML序列文件: {xml_file}"
            QMessageBox.information(self, "检测和导出完成", export_msg)

        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出过程中出错:\n{str(e)}")

    def handle_error(self, error_msg):
        self.result_text.append(f"错误: {error_msg}")
        self.status_label.setText("检测出错")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)

    def display_results(self):
        if not self.detection_results:
            return

        self.result_text.clear()
        self.result_text.append("检测结果:")
        for i, change in enumerate(self.detection_results, 1):
            self.result_text.append(
                f"{i}. 时间点: {change['time']} - 类型: {change['type']}"
            )

    def update_status(self, message):
        # 检查是否是检测到变化的消息
        if "检测到" in message and "变化" in message:
            # 添加时间戳
            timestamp = datetime.now().strftime("%H:%M:%S")
            message = f"[{timestamp}] {message}"
            # 将检测结果显示在文本框中
            self.result_text.append(message)
            # 滚动到最新内容
            self.result_text.moveCursor(QTextCursor.MoveOperation.End)
        else:
            # 进度信息显示在状态栏
            self.status_label.setText(message)
