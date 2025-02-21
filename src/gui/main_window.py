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
        
        # 统计检测结果
        total_changes = len(results)
        sudden_changes = sum(1 for r in results if r.get('type') == 'sudden')
        static_changes = sum(1 for r in results if r.get('type') == 'static')
        
        # 计算视频总时长
        if results:
            last_time = results[-1]['time'].total_seconds()
            hours = int(last_time // 3600)
            minutes = int((last_time % 3600) // 60)
            seconds = int(last_time % 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # 计算平均间隔
            avg_interval = last_time / (total_changes - 1) if total_changes > 1 else 0
            avg_minutes = int(avg_interval // 60)
            avg_seconds = int(avg_interval % 60)
        
        # 显示检测结果
        self.result_text.clear()
        self.result_text.append("🎉 检测完成！\n")
        self.result_text.append("📊 统计信息:")
        self.result_text.append(f"• 视频总时长: {duration_str}")
        self.result_text.append(f"• 检测到幻灯片变化: {total_changes} 处")
        self.result_text.append(f"  - 快速切换: {sudden_changes} 处")
        self.result_text.append(f"  - 渐变过渡: {static_changes} 处")
        if total_changes > 1:
            self.result_text.append(f"• 平均幻灯片停留时间: {avg_minutes}分{avg_seconds}秒\n")
        
        self.result_text.append("⏱️ 详细时间点:")
        
        # 显示每个变化点的详细信息
        for i, change in enumerate(results, 1):
            time_str = change['time'].strftime("%H:%M:%S.%f")[:-4]
            change_type = "快速切换" if change.get('type') == 'sudden' else "渐变过渡"
            
            # 获取更多变化细节
            details = []
            if 'mean_magnitude' in change:
                if change['mean_magnitude'] > 1.5:
                    details.append("大幅变化")
                elif change['mean_magnitude'] > 0.8:
                    details.append("中等变化")
                else:
                    details.append("轻微变化")
                
            if 'is_lecturer_motion' in change and change['is_lecturer_motion']:
                details.append("讲师活动")
            
            detail_str = f"({', '.join(details)})" if details else ""
            
            self.result_text.append(f"{i}. {time_str} - {change_type} {detail_str}")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("检测完成")

        # 自动导出文件
        try:
            video_dir = os.path.dirname(self.video_path)
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]

            # 导出CSV和XML
            csv_path = os.path.join(video_dir, f"{base_name}_markers.csv")
            xml_path = os.path.join(video_dir, f"{base_name}.xml")
            
            from ..exporters.csv_exporter import export_pr_markers
            from ..exporters.xml_exporter import export_to_fcpxml
            
            csv_file = export_pr_markers(self.detection_results, self.video_path, csv_path)
            xml_file = export_to_fcpxml(self.detection_results, self.video_path, xml_path)

            # 显示导出结果
            export_msg = "✨ 检测完成！\n\n"
            export_msg += "📁 导出文件:\n"
            if csv_file:
                export_msg += f"• CSV标记文件: {csv_file}\n"
            if xml_file:
                export_msg += f"• XML序列文件: {xml_file}\n\n"
            
            export_msg += "📊 检测统计:\n"
            export_msg += f"• 视频时长: {duration_str}\n"
            export_msg += f"• 总计检测到 {total_changes} 处变化\n"
            export_msg += f"  - 快速切换: {sudden_changes} 处\n"
            export_msg += f"  - 渐变过渡: {static_changes} 处\n"
            if total_changes > 1:
                export_msg += f"• 平均间隔: {avg_minutes}分{avg_seconds}秒\n\n"
            
            export_msg += "💡 提示:\n"
            export_msg += "1. 在PR中先导入视频文件\n"
            export_msg += "2. 然后导入XML文件\n"
            export_msg += "3. 如果提示视频脱机，请手动重新链接"
            
            QMessageBox.information(self, "检测和导出完成", export_msg)

        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出过程中出错:\n{str(e)}")

    def handle_error(self, error_msg):
        self.result_text.append(f"错误: {error_msg}")
        self.status_label.setText("检测出错")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)

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
