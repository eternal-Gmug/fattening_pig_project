import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QFormLayout,
                              QLineEdit, QComboBox, QColorDialog, QTabWidget, QSplitter, QCheckBox, QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer, QRect, QPoint, QEvent
from PySide6.QtGui import QPixmap, QColor, QFont, QImage, QPainter, QPen, QMouseEvent
 
 
class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频标注工具")
        self.resize(1200, 800)
        # 视频帧相关变量
        self.video_frames = []         # 存储所有视频帧
        self.video_path = ""           # 视频的输入路径
        self.current_frame_index = 0   # 当前显示的帧索引
        self.total_frame_count = 0     # 视频总帧数
        self.frame_rate = 30           # 视频的帧率
        self.rectangles = []  # 存储所有矩形框（原始图像坐标）
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.color = QColor(255, 0, 0)  # 默认红色
        self.thickness = 2
        self.current_pixmap = None     # 当前显示的图像（缩放后的QPixmap）
        self.original_image = None     # 原始图像（QImage）
 
        # 初始化配置
        self.config = {
            'output_json_path': './output',
            'output_video_path': './processed_videos',
            'background_color': '#f0f0f0',
            'frame_rate': 30,
            'model_path': ''
        }
 
        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
 
        # 设置样式
        self.set_style()
 
        # 创建菜单栏
        self.create_menu_bar()
 
        # 创建工具栏
        self.create_tool_bar()
 
        # 创建主内容区域
        self.create_main_content()
 
        # 创建状态栏
        self.statusBar().showMessage("就绪")
 
    def set_style(self):
        """设置应用程序样式"""
        self.setStyleSheet(""".QMainWindow { background-color: %s; }
                           QPushButton { padding: 4px 8px; border-radius: 4px; border: 1px solid black; }
                           QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; padding: 10px; }
                           QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; top: -1px; }
                           QTabBar::tab { padding: 6px 12px; margin-right: 2px; }
                           QTabBar::tab:selected { border-bottom: 2px solid #4a90e2; }""" % self.config['background_color'])
 
    def create_menu_bar(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        load_video_action = file_menu.addAction("加载视频")
        load_video_action.triggered.connect(self.load_video)
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)
 
    def create_tool_bar(self):
        """创建工具栏"""
        tool_bar = self.addToolBar("工具")
        tool_bar.setMovable(False)
        load_video_btn = QPushButton("加载视频")
        load_video_btn.setStyleSheet("QPushButton { padding: 4px 8px; border: none;}")
        load_video_btn.clicked.connect(self.load_video)
        tool_bar.addWidget(load_video_btn)
 
    def create_main_content(self):
        """创建主内容区域"""
        main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(main_splitter)
 
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
 
        # 视频信息
        info_group = QGroupBox("视频信息")
        info_layout = QVBoxLayout(info_group)
        top_info_layout = QHBoxLayout()
        self.video_id_label = QLabel("视频名称:")
        self.video_id_value = QLabel("")
        self.frame_num_label = QLabel("帧序号:")
        self.frame_num_value = QLabel("0/0")
        top_info_layout.addWidget(self.video_id_label)
        top_info_layout.addWidget(self.video_id_value)
        top_info_layout.addStretch()
        top_info_layout.addWidget(self.frame_num_label)
        top_info_layout.addWidget(self.frame_num_value)
        info_layout.addLayout(top_info_layout)
        left_layout.addWidget(info_group)
 
        # 视频显示区域
        self.video_display = QLabel("视频显示区域")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: #000000; color: white;")
        left_layout.addWidget(self.video_display)
        self.video_display.setMouseTracking(True)
        self.video_display.installEventFilter(self)  # 安装事件过滤器
 
        main_splitter.addWidget(left_panel)
 
        # 右侧面板
        right_panel = QTabWidget()
        main_splitter.addWidget(right_panel)
 
        # 标注控制标签页
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton("上一帧")
        self.next_frame_btn = QPushButton("下一帧")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        control_layout.addWidget(self.prev_frame_btn)
        control_layout.addWidget(self.next_frame_btn)
        control_group.setLayout(control_layout)
        annotation_layout.addWidget(control_group)
        annotation_layout.addStretch()
        right_panel.addTab(annotation_tab, "标注")
 
        main_splitter.setSizes([800, 400])
 
    def eventFilter(self, obj, event):
        """事件过滤器，处理鼠标事件"""
        if obj == self.video_display:
            if event.type() == QEvent.MouseButtonPress:
                self._handle_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                self._handle_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                self._handle_mouse_release(event)
            elif event.type() == QEvent.Paint:
                self._handle_paint()
        return super().eventFilter(obj, event)
 
    def _handle_mouse_press(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.original_image:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
 
    def _handle_mouse_move(self, event):
        """鼠标移动事件（实时预览）"""
        if self.drawing and self.original_image:
            self.end_point = event.pos()
            self.video_display.update()  # 触发重绘
 
    def _handle_mouse_release(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.drawing and self.original_image:
            self.drawing = False
            self.end_point = event.pos()
            # 保存矩形框（转换为原始图像坐标）
            if abs(self.start_point.x() - self.end_point.x()) > 5 and abs(self.start_point.y() - self.end_point.y()) > 5:
                rect = self._scale_rect_to_image(self.start_point, self.end_point)
                self.rectangles.append(rect)
            self.video_display.update()
 
    def _handle_paint(self):
        """绘制标注"""
        if not self.current_pixmap:
            return
 
        # 在QPixmap上绘制
        painter = QPainter(self.video_display.pixmap())
        pen = QPen(self.color, self.thickness)
        painter.setPen(pen)
 
        # 绘制所有已保存的矩形框
        for rect in self.rectangles:
            start, end = self._scale_rect_to_display(rect)
            painter.drawRect(QRect(start, end))
 
        # 绘制当前正在绘制的矩形（预览）
        if self.drawing and self.start_point and self.end_point:
            painter.drawRect(QRect(self.start_point, self.end_point))
 
        painter.end()
 
    def _scale_point_to_image(self, display_point):
        """将显示坐标转换为原始图像坐标"""
        if not self.current_pixmap or not self.original_image:
            return display_point
 
        display_size = self.video_display.size()
        pixmap_size = self.current_pixmap.size()
        scale_x = self.original_image.width() / pixmap_size.width()
        scale_y = self.original_image.height() / pixmap_size.height()
 
        # 考虑KeepAspectRatio的对齐方式
        aspect_ratio = min(
            display_size.width() / pixmap_size.width(),
            display_size.height() / pixmap_size.height()
        )
        scaled_width = pixmap_size.width() * aspect_ratio
        scaled_height = pixmap_size.height() * aspect_ratio
        offset_x = (display_size.width() - scaled_width) / 2
        offset_y = (display_size.height() - scaled_height) / 2
 
        x = (display_point.x() - offset_x) * scale_x
        y = (display_point.y() - offset_y) * scale_y
        return QPoint(int(x), int(y))
 
    def _scale_rect_to_image(self, start, end):
        """将显示坐标的矩形框转换为原始图像坐标"""
        p1 = self._scale_point_to_image(start)
        p2 = self._scale_point_to_image(end)
        return (p1, p2)
 
    def _scale_rect_to_display(self, image_rect):
        """将原始图像坐标的矩形框转换为显示坐标"""
        if not self.current_pixmap or not self.original_image:
            return (QPoint(), QPoint())
 
        start, end = image_rect
        display_size = self.video_display.size()
        pixmap_size = self.current_pixmap.size()
        scale_x = pixmap_size.width() / self.original_image.width()
        scale_y = pixmap_size.height() / self.original_image.height()
 
        # 考虑KeepAspectRatio的对齐方式
        aspect_ratio = min(
            display_size.width() / pixmap_size.width(),
            display_size.height() / pixmap_size.height()
        )
        scaled_width = pixmap_size.width() * aspect_ratio
        scaled_height = pixmap_size.height() * aspect_ratio
        offset_x = (display_size.width() - scaled_width) / 2
        offset_y = (display_size.height() - scaled_height) / 2
 
        x1 = start.x() * scale_x + offset_x
        y1 = start.y() * scale_y + offset_y
        x2 = end.x() * scale_x + offset_x
        y2 = end.y() * scale_y + offset_y
        return (QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2)))
 
    def load_video(self):
        """加载视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.video_path = file_path
            self.video_id_value.setText(file_path.split('/')[-1])
            self.load_video_frames()
 
    def load_video_frames(self):
        """提取视频帧"""
        self.video_frames = []
        self.current_frame_index = 0
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开视频文件")
            return
 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_frames.append(frame)
 
        cap.release()
        self.total_frame_count = len(self.video_frames)
        if self.video_frames:
            self.display_current_frame()
            self.update_frame_info()
 
    def display_current_frame(self):
        """显示当前帧"""
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index]
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            self.original_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.current_pixmap = QPixmap.fromImage(self.original_image)
            scaled_pixmap = self.current_pixmap.scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_display.setPixmap(scaled_pixmap)
 
    def update_frame_info(self):
        """更新帧信息"""
        self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
 
    def prev_frame(self):
        """上一帧"""
        if self.video_frames and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_current_frame()
            self.update_frame_info()
 
    def next_frame(self):
        """下一帧"""
        if self.video_frames and self.current_frame_index < len(self.video_frames) - 1:
            self.current_frame_index += 1
            self.display_current_frame()
            self.update_frame_info()
 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec())