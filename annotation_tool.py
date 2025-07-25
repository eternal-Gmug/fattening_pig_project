import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QFormLayout,
                              QLineEdit, QComboBox, QColorDialog, QTabWidget, QSplitter, QCheckBox, QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap, QColor, QFont, QImage


class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频标注工具")
        self.resize(1200, 800)
        # 视频帧相关变量
        self.video_frames = []      #存储所有视频帧
        self.video_path = ""        #视频的输入路径
        self.current_frame_index = 0   #当前显示的帧索引
        self.total_frame_count = 0     #视频总帧数
        self.frame_rate = 30           #视频的帧率

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

        # 文件菜单
        file_menu = menu_bar.addMenu("文件")
        load_video_action = file_menu.addAction("加载视频")
        load_video_action.triggered.connect(self.load_video)

        save_project_action = file_menu.addAction("保存项目")
        save_project_action.triggered.connect(self.save_project)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        # 编辑菜单
        edit_menu = menu_bar.addMenu("编辑")
        preferences_action = edit_menu.addAction("首选项")
        preferences_action.triggered.connect(self.open_preferences)

        # 处理菜单
        process_menu = menu_bar.addMenu("处理")
        run_model_action = process_menu.addAction("运行模型")
        run_model_action.triggered.connect(self.run_model)

        export_frames_action = process_menu.addAction("导出帧")
        export_frames_action.triggered.connect(self.export_frames)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)

    def create_tool_bar(self):
        """创建工具栏"""
        tool_bar = self.addToolBar("工具")
        tool_bar.setMovable(False)

        load_video_btn = QPushButton("加载视频")
        load_video_btn.setStyleSheet("QPushButton { padding: 4px 8px; border: none;}")
        load_video_btn.clicked.connect(self.load_video)
        tool_bar.addWidget(load_video_btn)

        tool_bar.addSeparator()

        run_model_btn = QPushButton("运行模型")
        run_model_btn.setStyleSheet("QPushButton { padding: 4px 8px;border: none;}")
        run_model_btn.clicked.connect(self.run_model)
        tool_bar.addWidget(run_model_btn)

        export_frames_btn = QPushButton("导出帧")
        export_frames_btn.setStyleSheet("QPushButton { padding: 4px 8px; border: none;}")
        export_frames_btn.clicked.connect(self.export_frames)
        tool_bar.addWidget(export_frames_btn)


    def create_main_content(self):
        """创建主内容区域"""
        # 创建分割器
        main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(main_splitter)

        # 左侧面板 - 视频和标注显示
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # 添加视频信息显示区域
        info_group = QGroupBox("视频信息")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(10, 10, 10, 10)

        # 设置字体
        font = QFont()
        font.setPointSize(10)
        info_group.setFont(font)

        # 第一行：视频编号、帧序号、帧率
        top_info_layout = QHBoxLayout()
        
        self.video_id_label = QLabel("视频编号: ")
        self.video_id_label.setFont(font)
        self.video_id_value = QLabel("001")
        self.video_id_value.setFont(font)
        
        self.frame_num_label = QLabel("帧序号: ")
        self.frame_num_label.setFont(font)
        self.frame_num_value = QLabel(str(self.current_frame_index) + "/" + str(self.total_frame_count))
        self.frame_num_value.setFont(font)
        
        self.fps_label = QLabel("帧率: ")
        self.fps_label.setFont(font)
        self.fps_value = QLabel(str(self.config['frame_rate']))
        self.fps_value.setFont(font)
        
        top_info_layout.addWidget(self.video_id_label)
        top_info_layout.addWidget(self.video_id_value)
        top_info_layout.addSpacing(15)
        top_info_layout.addWidget(self.frame_num_label)
        top_info_layout.addWidget(self.frame_num_value)
        top_info_layout.addSpacing(15)
        top_info_layout.addWidget(self.fps_label)
        top_info_layout.addWidget(self.fps_value)
        top_info_layout.addStretch()
        
        # 第二行：是否为起始帧
        start_frame_layout = QHBoxLayout()
        
        self.start_frame_label = QLabel("是否为起始帧: ")
        self.start_frame_label.setFont(font)
        self.start_frame_checkbox = QCheckBox()
        
        start_frame_layout.addWidget(self.start_frame_label)
        start_frame_layout.addWidget(self.start_frame_checkbox)
        start_frame_layout.addStretch()
        
        info_layout.addLayout(top_info_layout)
        info_layout.addLayout(start_frame_layout)
        left_layout.addWidget(info_group)

        # 视频显示区域
        self.video_display = QLabel("视频显示区域")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: #000000; color: white;")
        left_layout.addWidget(self.video_display)

        # 移除视频控制部分代码

        main_splitter.addWidget(left_panel)

        # 右侧面板 - 控制和设置
        right_panel = QTabWidget()
        main_splitter.addWidget(right_panel)

        # 标注控制标签页
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout(annotation_tab)

        # 标注工具组
        tool_group = QGroupBox("标注工具")
        tool_layout = QHBoxLayout()

        self.rect_tool_btn = QPushButton("矩形")
        self.polygon_tool_btn = QPushButton("多边形")
        self.point_tool_btn = QPushButton("点")
        self.text_tool_btn = QPushButton("文本")

        tool_layout.addWidget(self.rect_tool_btn)
        tool_layout.addWidget(self.polygon_tool_btn)
        tool_layout.addWidget(self.point_tool_btn)
        tool_layout.addWidget(self.text_tool_btn)

        tool_group.setLayout(tool_layout)
        annotation_layout.addWidget(tool_group)

        # 标注属性组
        props_group = QGroupBox("标注属性")
        props_layout = QFormLayout()

        self.label_input = QLineEdit()
        self.color_btn = QPushButton("选择颜色")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 10)
        self.thickness_slider.setValue(2)

        props_layout.addRow("标签:", self.label_input)
        props_layout.addRow("颜色:", self.color_btn)
        props_layout.addRow("线宽:", self.thickness_slider)

        props_group.setLayout(props_layout)
        annotation_layout.addWidget(props_group)

        annotation_layout.addStretch()
        right_panel.addTab(annotation_tab, "标注")

        # 模型设置标签页
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        model_group = QGroupBox("模型设置")
        model_form_layout = QFormLayout()

        self.model_path_input = QLineEdit(self.config['model_path'])
        self.browse_model_btn = QPushButton("浏览...")
        self.conf_threshold_slider = QSlider(Qt.Horizontal)
        self.conf_threshold_slider.setRange(0, 100)
        self.conf_threshold_slider.setValue(50)
        self.iou_threshold_slider = QSlider(Qt.Horizontal)
        self.iou_threshold_slider.setRange(0, 100)
        self.iou_threshold_slider.setValue(30)

        model_form_layout.addRow("模型路径:", self.model_path_input)
        model_form_layout.addRow("", self.browse_model_btn)
        model_form_layout.addRow("置信度阈值 (%):", self.conf_threshold_slider)
        model_form_layout.addRow("IOU阈值 (%):", self.iou_threshold_slider)

        model_group.setLayout(model_form_layout)
        model_layout.addWidget(model_group)

        model_layout.addStretch()
        right_panel.addTab(model_tab, "模型")

        # 输出设置标签页
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)

        output_group = QGroupBox("输出设置")
        output_form_layout = QFormLayout()

        self.output_json_path_input = QLineEdit(self.config['output_json_path'])
        self.output_json_path_input.setReadOnly(True)
        self.browse_output_btn = QPushButton("更改输出Json路径")
        self.browse_output_btn.setStyleSheet("border : 1px solid black;")
        self.browse_output_btn.clicked.connect(self.browse_output_json_path)

        self.output_video_path_input = QLineEdit(self.config['output_video_path'])
        self.output_video_path_input.setReadOnly(True)
        self.browse_output_video_btn = QPushButton("更改输出视频路径")
        self.browse_output_video_btn.clicked.connect(self.browse_output_video_path)
        
        '''
        self.frame_format_combo = QComboBox()
        self.frame_format_combo.addItems(['jpg', 'png', 'bmp'])
        self.frame_quality_slider = QSlider(Qt.Horizontal)
        self.frame_quality_slider.setRange(0, 100)
        self.frame_quality_slider.setValue(90)
        '''

        output_form_layout.addRow("输出json路径:", self.output_json_path_input)
        output_form_layout.addRow("", self.browse_output_btn)
        output_form_layout.addRow("输出视频路径:", self.output_video_path_input)
        output_form_layout.addRow("", self.browse_output_video_btn)

        '''
        output_form_layout.addRow("帧格式:", self.frame_format_combo)
        output_form_layout.addRow("图像质量 (%):", self.frame_quality_slider)
        '''

        output_group.setLayout(output_form_layout)
        output_layout.addWidget(output_group)

        output_layout.addStretch()
        right_panel.addTab(output_tab, "输出")

        # 设置分割器初始大小
        main_splitter.setSizes([800, 400])

    def load_video(self):
        """加载视频文件"""
        # 从本地文件夹加载视频
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.video_path = file_path
            self.load_video_frames()
    
    def load_video_frames(self):
        """按帧率从视频中提取帧"""
        # 清空之前的帧数据
        self.video_frames = []
        self.current_frame_index = 0

        # 使用OpenCV读取视频
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            QMessageBox.warning(self, "警告", "无法打开视频文件")
            return
        
        # 获取视频帧率
        self.frame_rate = capture.get(cv2.CAP_PROP_FPS)

        # 读取所有帧
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            # 转换BGR为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_frames.append(frame_rgb)

        capture.release()

        # 更新界面显示
        if self.video_frames:
            self.display_current_frame()
            # 更新帧序号显示
            self.update_frame_info()
        else:
            QMessageBox.warning(self, "警告", "视频中未提取到帧")

    def display_current_frame(self):
        """显示当前帧"""
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index]
            # 转换为QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_frame_info(self):
        """更新帧信息显示"""
        self.total_frame_count = len(self.video_frames)
        self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
        #TODO：是否为初始帧

    def save_project(self):
        """保存项目"""
        # 具体实现代码暂不生成
        pass

    def open_preferences(self):
        """打开首选项设置"""
        # 具体实现代码暂不生成
        pass

    def run_model(self):
        """运行模型处理视频"""
        # 具体实现代码暂不生成
        pass

    def export_frames(self):
        """导出视频帧"""
        # 具体实现代码暂不生成
        pass

    def show_about(self):
        """显示关于对话框"""
        # 具体实现代码暂不生成
        pass

    # 浏览并选择输出JSON路径
    def browse_output_json_path(self):
        """浏览并选择输出JSON路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择输出JSON文件", self.config['output_json_path'], "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.config['output_json_path'] = file_path
            self.output_json_path_input.setText(file_path)

    # 浏览并选择输出视频路径
    def browse_output_video_path(self):
        """浏览并选择输出视频路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择输出视频文件", self.config['output_video_path'], "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            self.config['output_video_path'] = file_path
            self.output_video_path_input.setText(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec())