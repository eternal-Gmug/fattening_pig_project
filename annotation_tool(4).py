import sys
import cv2
import json
import os
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QFormLayout,
                              QLineEdit, QComboBox, QColorDialog, QTabWidget, QSplitter, QCheckBox, QMessageBox,QSizePolicy)
from PySide6.QtCore import Qt, QSize, QTimer, QEvent,QPoint
from PySide6.QtGui import QPixmap, QColor, QFont, QImage
from ultralytics import YOLO

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频标注工具")
        self.resize(1200, 800)
        # 视频帧相关变量
        self.video_frames = []         #存储所有视频帧
        self.video_path = ""           #视频的输入路径
        self.current_frame_index = 0   #当前显示的帧索引
        self.total_frame_count = 0     #视频总帧数
        self.frame_rate = 30           #视频的帧率
        self.key_frames = {}           #存储视频的关键帧索引
        # 标注相关变量
        self.annotations = {}          # 存储所有标注信息
        self.annotation_widgets = {}   # 存储所有标注框的文本输入框{frameindex:[{annotationid:widget},{annotationid:widget}]}
        self.current_tool = None       # 当前选中的标注工具
        self.drawing = False           # 是否正在绘制标注
        self.start_point = None        # 绘制开始点
        self.end_point = None          # 绘制结束点
        self.current_color = QColor(Qt.red) # 当前标注的颜色（默认为红色）
        self.current_thickness = 2     # 当前标注的线宽
        self.dragging = False          # 是否正在拖动标注
        self.dragging_annotation = None # 当前正在拖动的标注
        self.drag_offset = QPoint()    # 拖动偏移量

        # 初始化配置
        self.config = {
            'default_video_path': './input_videos',
            'output_txt_path': './output',
            'output_video_path': './processed_videos',
            'background_color': '#f0f0f0',
            'model_path': './yolov8n.pt'
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
        export_frames_action.triggered.connect(self.export_annotated_video)

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
        export_frames_btn.clicked.connect(self.export_annotated_video)
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
        
        self.video_id_label = QLabel("视频名称: ")
        self.video_id_label.setFont(font)
        self.video_id_value = QLabel("")
        self.video_id_value.setFont(font)
        
        self.frame_num_label = QLabel("帧序号: ")
        self.frame_num_label.setFont(font)
        self.frame_num_value = QLabel(str(self.current_frame_index) + "/" + str(self.total_frame_count))
        self.frame_num_value.setFont(font)
        
        self.fps_label = QLabel("帧率: ")
        self.fps_label.setFont(font)
        self.fps_value = QLabel(str(self.frame_rate))
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
        
        # 第二行：是否为关键帧
        essential_frame_layout = QHBoxLayout()
        
        self.essential_frame_label = QLabel("是否为关键帧: ")
        self.essential_frame_label.setFont(font)
        # 关键帧复选框能够根据当前帧是否为关键帧进行切换
        self.essential_frame_checkbox = QCheckBox()
        self.essential_frame_checkbox.setChecked(self.key_frames.get(self.current_frame_index, False))
        # 关键帧复选框的状态改变时，更新当前帧的关键帧状态，并保存每一帧的关键帧状态
        self.essential_frame_checkbox.stateChanged.connect(self.update_essential_frame_state)

        essential_frame_layout.addWidget(self.essential_frame_label)
        essential_frame_layout.addWidget(self.essential_frame_checkbox)
        essential_frame_layout.addStretch()
        
        info_layout.addLayout(top_info_layout)
        info_layout.addLayout(essential_frame_layout)
        left_layout.addWidget(info_group)

        # 视频显示区域
        TODO: 视频显示区域在界面可缩放时保持原视频比例
        self.video_display = QLabel("视频显示区域")
        # 居中文本
        self.video_display.setAlignment(Qt.AlignCenter)

        self.video_display.setMinimumSize(800,450)      # 初始比例按业务需求决定

        self.video_display.setStyleSheet("background-color: #000000; color: white;")

         # 设置尺寸策略，初始尺寸为700*525，可以根据软件窗口的大小进行调整
        self.video_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 将视频显示区域居中放置在左侧布局中
        left_layout.addWidget(self.video_display)

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

        self.mouse_btn = QPushButton("鼠标拖动")
        self.rect_tool_btn = QPushButton("矩形")
        self.point_tool_btn = QPushButton("点")

        tool_layout.addWidget(self.mouse_btn)
        tool_layout.addWidget(self.rect_tool_btn)
        tool_layout.addWidget(self.point_tool_btn)

        tool_group.setLayout(tool_layout)
        annotation_layout.addWidget(tool_group)

        # 标注属性组
        props_group = QGroupBox("标注属性")
        props_layout = QFormLayout()

        #self.label_input = QLineEdit()
        self.color_btn = QPushButton("选择颜色")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 10)
        self.thickness_slider.setValue(2)

        #props_layout.addRow("标签:", self.label_input)
        props_layout.addRow("颜色:", self.color_btn)
        props_layout.addRow("线宽:", self.thickness_slider)

        props_group.setLayout(props_layout)
        annotation_layout.addWidget(props_group)

        TODO: 添加标注框列表滚动区域在多标注下需要滚动
        self.annotations_group = QGroupBox("标注框列表")
        self.annotations_layout = QVBoxLayout()
        self.annotations_group.setLayout(self.annotations_layout)
        annotation_layout.addWidget(self.annotations_group)

        #添加重置该标签、上一帧、下一帧的按钮组
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重置该标签")
        self.prev_frame_btn = QPushButton("上一帧")
        self.next_frame_btn = QPushButton("下一帧")

        # 连接按钮点击事件
        self.rect_tool_btn.clicked.connect(lambda: self.set_current_tool('rectangle'))
        # 连接颜色选择按钮
        self.color_btn.clicked.connect(self.select_color)
        # 连接点工具按钮点击事件
        self.point_tool_btn.clicked.connect(lambda: self.set_current_tool('point'))
        # 连接鼠标拖动按钮点击事件
        self.mouse_btn.clicked.connect(lambda: self.set_current_tool('mouse'), self.mouse_drag)

        self.reset_btn.clicked.connect(self.reset_annotation)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)

        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.prev_frame_btn)
        control_layout.addWidget(self.next_frame_btn)

        control_group.setLayout(control_layout)
        annotation_layout.addWidget(control_group)

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

        self.output_txt_path_input = QLineEdit(self.config['output_txt_path'])
        self.output_txt_path_input.setReadOnly(True)
        self.browse_output_btn = QPushButton("更改输出txt路径")
        self.browse_output_btn.setStyleSheet("border : 1px solid black;")
        self.browse_output_btn.clicked.connect(self.browse_output_txt_path)

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

        output_form_layout.addRow("输出txt路径:", self.output_txt_path_input)
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
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        # 确保视频显示区域居中
        self.video_display.parent().layout().setAlignment(self.video_display, Qt.AlignCenter)

    # 从本地文件夹加载视频
    def load_video(self):
        """加载视频文件"""
        default_path = self.config.get('default_video_path', './input_videos')
        video_extensions = ('.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png', '.bmp')
        
        # 确保默认目录存在
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        
        # 获取默认目录中的视频文件
        video_files = []
        if os.path.isdir(default_path):
            video_files = [f for f in os.listdir(default_path) 
                          if os.path.isfile(os.path.join(default_path, f)) 
                          and f.lower().endswith(video_extensions)]
        
        file_path = None
        
        # 根据视频文件数量决定操作
        if len(video_files) == 1:
            # 自动加载唯一视频
            file_path = os.path.join(default_path, video_files[0])
        elif len(video_files) > 1:
            # 多个视频，提示用户手动选择
            QMessageBox.information(self, "多个视频文件", f"默认目录 '{default_path}' 中找到多个视频文件，请手动选择。")
        # else: 没有视频文件，继续到文件对话框
        
        # 如果没有自动获取到文件路径，则显示文件对话框
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频或图片文件", default_path, "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp);;All Files (*)"
            )
        
        if file_path:
            self.video_path = file_path
            self.video_id_value.setText(os.path.basename(file_path))  # 使用basename更准确
            self.load_video_frames()
    
    # 将视频切换成图片帧
    def load_video_frames(self):
        """按帧率从视频中提取帧"""
        # 清空之前的帧数据
        self.video_frames = []
        self.current_frame_index = 0
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开视频文件")
                return
            # 读取视频信息
            self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            TODO: 这里需要导入yolov8模型实现视频对视频帧的识别
            # 加载yolov8模型
            try:
                self.model = YOLO("yolov8n.pt")
                # 可以加载自定义模型
            except Exception as e:
                QMessageBox.warning(self, "模型加载错误", f"无法加载yolov8模型: {e}\n将使用空检测结果")
                print(e)
                self.model = None
                
            # 按帧率提取帧
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 存储原始帧
                self.video_frames.append(frame)

                # 使用yolov8模型进行识别
                if self.model is not None:
                    # 只检测猪类（COCO数据集中猪的类别ID是13），这里是单帧图像
                    results = self.model.predict(frame, classes=[13], conf=0.2)
                    # 处理检测结果
                    frame_annotation = []
                    boxs = results[0].boxes
                    for i,box in enumerate(boxs):
                        # 获取标注框的坐标
                        x1,y1,x2,y2 = map(int,box.xyxy[0])
                        # 获取置信度
                        confidence = float(box.conf[0])
                        # 构建标注信息
                        annotation ={
                            'id': i + 1,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'confidence': confidence,
                            'text': '',
                            'color':(0,255,0)
                        }
                        frame_annotation.append(annotation)
                    
                    # 保存标注信息
                    if frame_annotation:
                        self.annotations[frame_idx] = frame_annotation
                    
                    frame_idx += 1

            cap.release()
            # 更新界面显示
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_info()
                # 如果有模型检测结果，显示提示
                if self.annotations:
                    #detected_count = sum(len(anns) for anns in self.annotations.values())
                    #QMessageBox.information(self, "检测结果", f"检测到 {detected_count} 只猪")
                    QMessageBox.information("检测完成")
            else:
                QMessageBox.warning(self, "警告", "视频中未提取到帧")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载视频帧时出错: {e}")
            print(e)
            return

    # 显示当前图片帧
    def display_current_frame(self):
        """显示当前帧，并绘制标注"""
        TODO:可以直接从已经被yolov8处理过的帧提取结果
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index].copy()
            # 转换为QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # 绘制已有的标注（包含yolov8识别结果）
            if self.current_frame_index in self.annotations:
                for annotation in self.annotations[self.current_frame_index]:
                    # 绘制矩形
                    color = annotation['color']
                    cv2.rectangle(frame, (annotation['x1'], annotation['y1']), 
                                  (annotation['x2'], annotation['y2']), color, 2)
                    # 绘制标签
                    label_text = f"{annotation['id']}"
                    cv2.putText(frame, label_text, (annotation['x1'], annotation['y1']-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制正在绘制的矩形
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(frame, (self.start_point.x(), self.start_point.y()), 
                              (self.end_point.x(), self.end_point.y()), 
                              (self.current_color.red(), self.current_color.green(), self.current_color.blue()), 2)
            
            # 绘制正在移动的标注框
            if self.dragging_annotation:
                cv2.rectangle(frame, (self.dragging_annotation['x1'], self.dragging_annotation['y1']), 
                              (self.dragging_annotation['x2'], self.dragging_annotation['y2']), 
                              (self.current_color.red(), self.current_color.green(), self.current_color.blue()), 2)

            # 获取视频显示区域的大小
            display_width = self.video_display.width()
            display_height = self.video_display.height()

            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            scaled_image = q_image.scaled(display_width,display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.video_display.setPixmap(QPixmap.fromImage(scaled_image))

            # 更新标注组件
            self.update_annotation_widgets()

    # 更新当前帧的图片信息（视频名称、总帧数、当前帧数、是否为关键帧、fps）
    def update_frame_info(self):
        """更新帧信息显示"""
        self.video_id_value.setText(str(self.video_path.split('/')[-1]).split('.')[0])
        self.total_frame_count = len(self.video_frames)
        self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
        self.fps_value.setText(str(self.frame_rate))
        self.essential_frame_checkbox.setChecked(self.key_frames.get(self.current_frame_index, False))

    # 保存项目标注信息到TXT文件
    def save_project(self):
        """保存标注信息到TXT文件"""
        # 获取视频名称作为文件夹名
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        # 确保输出目录存在，使用视频名称作为子文件夹
        output_dir = os.path.join(self.config['output_txt_path'], video_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 准备保存的数据
        project_data = {
            'video_path': self.video_path,
            'frame_rate': self.frame_rate,
            'total_frames': self.total_frames,
            'annotations': self.annotations
        }

        # 保存到TXT文件，每个帧一个文件
        try:
            for frame_idx, annotations in project_data['annotations'].items():
                # 为每个帧创建单独的TXT文件，文件名即为帧索引
                txt_file_name = f"{frame_idx}.txt"
                txt_file_path = os.path.join(output_dir, txt_file_name)
                
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    for ann in annotations:
                        x_min, y_min = ann['x1'], ann['y1']
                        x_max, y_max = ann['x2'], ann['y2']
                        # 获取原始视频帧的尺寸（从第一帧获取）
                        if self.video_frames:
                            frame_height, frame_width = self.video_frames[0].shape[:2]
                            # 计算归一化坐标
                            x_center = (x_min + x_max) / 2 / frame_width
                            y_center = (y_min + y_max) / 2 / frame_height
                            width = (x_max - x_min) / frame_width
                            height = (y_max - y_min) / frame_height
                        else:
                            x_center = 0
                            y_center = 0
                            width = 0
                            height = 0
                        f.write(f"{0},{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f}\n")
            QMessageBox.information(self, "成功", f"所有帧已保存到 {output_dir}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存项目时出错: {e}")

    def open_preferences(self):
        """打开首选项设置"""
        # 具体实现代码暂不生成
        pass

    def run_model(self):
        """运行模型处理视频"""
        # 具体实现代码暂不生成
        pass

    def export_annotated_video(self):
        """导出带有标注的视频帧"""
        if not self.video_frames:
            QMessageBox.warning(self, "警告", "请先加载视频")
            return
        
        try:
            # 获取视频信息
            height, width, _ = self.video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.config['output_video_path'], fourcc, self.frame_rate, (width, height))

            # 逐帧处理
            for i, frame in enumerate(self.video_frames):
                # 复制帧
                annotated_frame = frame.copy()
                # 绘制标注
                if i in self.annotations:
                    for annotation in self.annotations[i]:
                        color = annotation['color']
                        # 绘制矩形
                        cv2.rectangle(annotated_frame, (annotation['x1'], annotation['y1']), 
                                      (annotation['x2'], annotation['y2']), color, 2)
                        # 绘制标签
                        label_text = f"{annotation['id']}. {annotation['label']}"
                        cv2.putText(annotated_frame, label_text, (annotation['x1'], annotation['y1']-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        # 绘制文本信息
                        if annotation['text']:
                            text_text = f"{annotation['text']}"
                            cv2.putText(annotated_frame, text_text, (annotation['x1'], annotation['y2']+20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 转换为BGR格式并写入视频
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            # 释放资源
            out.release()
            QMessageBox.information(self, "成功", f"带标注的视频已导出到 {self.config['output_video_path']}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出视频时出错: {str(e)}")

    def show_about(self):
        """显示关于对话框"""
        # 具体实现代码暂不生成
        pass

    # 更新当前帧的关键帧状态
    def update_essential_frame_state(self):
        """更新当前帧的关键帧状态"""
        # 获取当前帧的关键帧状态
        is_essential = self.essential_frame_checkbox.isChecked()
        # 更新当前帧的关键帧状态
        self.key_frames[self.current_frame_index] = is_essential

    # 设置当前标注工具
    def set_current_tool(self,tool_type):
        """设置当前标注工具"""
        self.current_tool = tool_type
        # 更新按钮样式以显示当前选中的样式
        self.mouse_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'mouse' else "")
        self.rect_tool_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'rectangle' else "")
        self.point_tool_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'point' else "")
        # 启用视频显示区域的鼠标跟踪
        self.video_display.setMouseTracking(True)
        # 安装事件过滤器以捕获鼠标跟踪
        self.video_display.installEventFilter(self)

    # 选择颜色
    def select_color(self):
        """选择标注颜色"""
        color = QColorDialog.getColor(self.current_color, self, "选择标注颜色")
        if color.isValid():
            self.current_color = color

    # 事件过滤器,实现鼠标拖动标注框
    def eventFilter(self,obj,event):
        """事件过滤器，用于捕获视频显示区域的鼠标事件"""
        # 在图片显示区域下绘制标注框
        if obj is self.video_display and self.current_tool == 'rectangle':
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # 鼠标按下，开始绘制
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    self.drawing = True
                    self.start_point = mapped_point
                    self.end_point = mapped_point
                return True
            elif event.type() == QEvent.MouseMove and self.drawing:
                # 鼠标移动，更新结束点
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    self.end_point = mapped_point
                    # 重绘当前帧
                    self.display_current_frame()
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.drawing:
                # 鼠标释放，结束绘制
                self.drawing = False
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    self.end_point = mapped_point
                    # 确保起始点和结束点不同
                    if self.start_point != self.end_point:
                        self.add_annotation()
                # 重绘当前帧
                self.display_current_frame()
                return True
        
        # 在图片显示区域下实现鼠标拖动标注框
        if obj is self.video_display and self.current_tool == 'mouse':
            # 鼠标按下，检查是否在标注框内
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    # 检查当前帧是否有标注框
                    if self.current_frame_index in self.annotations:
                        self.is_mouse_in_annotation(mapped_point)
                return True
            # 鼠标移动，拖动标注框
            elif event.type() == QEvent.MouseMove and self.dragging:
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    # 计算新的标注框位置
                    self.mouse_drag(mapped_point)
                    # 重绘当前帧
                    self.display_current_frame()
                return True

            # 鼠标释放，结束拖动，更新标注框位置信息
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.dragging:
                self.dragging = False
                # 更新当前帧的标注框位置
                self.update_annotation_position()
                # 清除当前拖动的标注框
                self.dragging_annotation = None
                # 重绘当前帧
                self.display_current_frame()
                return True

        return super().eventFilter(obj, event)

    # 将显示窗口的坐标映射到原始视频帧的坐标
    def map_to_original_frame(self, point):
        """将显示窗口中的坐标映射到原始视频帧的坐标"""
        if not self.video_frames or self.current_frame_index < 0 or self.current_frame_index >= len(self.video_frames):
            return point
        
        # 获取原始视频帧的尺寸
        original_height, original_width = self.video_frames[self.current_frame_index].shape[:2]
        
        # 获取显示窗口的尺寸
        display_width = self.video_display.width()
        display_height = self.video_display.height()
        
        # 计算实际显示的视频尺寸（考虑宽高比）
        if original_width / original_height > display_width / display_height:
            # 视频宽度是限制因素
            display_video_width = display_width
            display_video_height = display_width * original_height / original_width
        else:
            # 视频高度是限制因素
            display_video_height = display_height
            display_video_width = display_height * original_width / original_height
        
        # 计算缩放比例
        scale = display_video_width / original_width
        
        # 计算显示窗口中的偏移量
        offset_x_display = (display_width - display_video_width) / 2
        offset_y_display = (display_height - display_video_height) / 2
        
        # 检查鼠标坐标是否在视频显示区域内
        if (point.x() < offset_x_display or point.x() >= offset_x_display + display_video_width or
            point.y() < offset_y_display or point.y() >= offset_y_display + display_video_height):
            # 鼠标在视频显示区域外，返回无效坐标
            return QPoint(-1, -1)
        
        # 调整鼠标坐标，减去显示窗口中的偏移量
        adjusted_x = point.x() - offset_x_display
        adjusted_y = point.y() - offset_y_display
        
        # 将调整后的坐标映射到原始视频帧坐标
        mapped_x = int(adjusted_x / scale)
        mapped_y = int(adjusted_y / scale)
        
        # 确保映射后的坐标在原始视频帧范围内
        mapped_x = max(0, min(mapped_x, original_width - 1))
        mapped_y = max(0, min(mapped_y, original_height - 1))
        
        return QPoint(mapped_x, mapped_y)

    # TODO:实现鼠标拖动标注框
    def mouse_drag(self,mapped_point):
        # 计算新的标注框位置
        new_x1 = mapped_point.x() - self.drag_offset.x()
        new_y1 = mapped_point.y() - self.drag_offset.y()
        width = self.dragging_annotation['x2'] - self.dragging_annotation['x1']
        height = self.dragging_annotation['y2'] - self.dragging_annotation['y1']
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height
        # 确保标注框不超出图片帧边界
        frame_height,frame_width = self.video_frames[self.current_frame_index].shape[:2]
        new_x1 = max(0, min(new_x1, frame_width - 1))
        new_y1 = max(0, min(new_y1, frame_height - 1))
        new_x2 = max(0, min(new_x2, frame_width - 1))
        new_y2 = max(0, min(new_y2, frame_height - 1))
        # 更新标注框位置
        self.dragging_annotation['x1'] = new_x1
        self.dragging_annotation['y1'] = new_y1
        self.dragging_annotation['x2'] = new_x2
        self.dragging_annotation['y2'] = new_y2

    # 判断当前鼠标是否在标注框内
    def is_mouse_in_annotation(self,mapped_point):
        for anno in self.annotations[self.current_frame_index]:
            x1, y1, x2, y2 = anno['x1'], anno['y1'], anno['x2'], anno['y2']
            if x1 <= mapped_point.x() <= x2 and y1 <= mapped_point.y() <= y2:
                self.dragging = True
                # 记录当前拖动的标注
                self.dragging_annotation = anno
                # 记录当前鼠标位置与左上角的偏移量
                self.drag_offset = mapped_point - QPoint(x1, y1)
                break

    # 添加标注框
    def add_annotation(self):
        """添加标注框"""
        # 确保当前帧有视频数据
        if not self.video_frames or self.current_frame_index < 0 or self.current_frame_index >= len(self.video_frames):
            return
        
        # 计算当前帧的最大标注id,从1开始递增
        max_id = 0
        if self.current_frame_index in self.annotations:
            max_id = max([anno['id'] for anno in self.annotations[self.current_frame_index]])
        annotation_id = max_id + 1

        # 计算矩形坐标（确保x1 < x2, y1 < y2）
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())

        # 获取标签文本
        label = f"标签{annotation_id}"

        # 存储标注信息
        if self.current_frame_index not in self.annotations:
            self.annotations[self.current_frame_index] = []
        self.annotations[self.current_frame_index].append({
            'id': annotation_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label,
            'text': '',
            'color': (self.current_color.red(), self.current_color.green(), self.current_color.blue()),
            'confidence': 1.0  # 手动标注默认置信度为1.0
        })

        # 添加文本输入框
        self.add_annotation_widget(annotation_id, label)

        # 清空标签输入框
        #self.label_input.clear()
        
    # 添加标注框的文本输入框
    def add_annotation_widget(self, annotation_id, label):
        """添加标注框的文本输入框"""
        # 确保当前帧有控件列表
        if self.current_frame_index not in self.annotation_widgets:
            self.annotation_widgets[self.current_frame_index] = []
        # 创建水平布局
        h_layout = QHBoxLayout()
        # 创建标签
        label_widget = QLabel(label)
        h_layout.addWidget(label_widget)
        # 创建文本输入框
        text_input = QLineEdit()
        text_input.setPlaceholderText(f"输入猪{annotation_id}的体重...")
        text_input.setObjectName(f"annotation_{annotation_id}")
        # 保存文本输入框引用
        self.annotation_widgets[self.current_frame_index].append({
            'id': annotation_id,
            'weight': text_input
        })
        # 将文本输入的内容加入标注信息的text中
        self.annotations[self.current_frame_index][annotation_id-1]['text'] = text_input

        # 连接文本变化事件并将
        text_input.textChanged.connect(lambda text, aid=annotation_id: self.update_annotation_text(aid, text))
        h_layout.addWidget(text_input)
        # 添加到布局
        self.annotations_layout.addLayout(h_layout)

    # 更新标注框的文本信息
    def update_annotation_text(self, annotation_id, text):
        """更新标注框的文本信息"""
        # 检查当前帧是否有标注
        if self.current_frame_index in self.annotations:
            for anno in self.annotations[self.current_frame_index]:
                if anno['id'] == annotation_id:
                    anno['weight'] = text
                    # 同步更新标注信息的text
                    self.annotations[self.current_frame_index][annotation_id-1]['text'] = text
                    break

    # 更新当前帧的标注框信息（拖动标注框）
    def update_annotation_position(self):
        """更新当前帧的标注框信息"""
        self.annotations[self.current_frame_index][self.dragging_annotation['id']-1] = self.dragging_annotation

    # 更新标注组件
    def update_annotation_widgets(self):
        """更新标注组件"""
        # 清除现有控件
        while self.annotations_layout.count() > 0:
            item = self.annotations_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                self.annotations_layout.removeWidget(widget)
            else:
                # 处理布局项
                layout = item.layout()
                if layout:
                    for i in range(layout.count()):
                        sub_item = layout.itemAt(i)
                        sub_widget = sub_item.widget()
                        if sub_widget:
                            sub_widget.hide()

        # 根据每一帧的控件信息添加当前帧的控件
        if self.current_frame_index in self.annotation_widgets:
            for item in self.annotation_widgets[self.current_frame_index]:
                text_input = item['weight']
                # 创建水平布局
                h_layout = QHBoxLayout()
                label_widget = QLabel(f"{self.get_annotation_label(item['id'])}:")
                h_layout.addWidget(label_widget)
                h_layout.addWidget(text_input)
                self.annotations_layout.addLayout(h_layout)
                # 显示控件
                label_widget.show()
                text_input.show()
    
    # 获取标注标签
    def get_annotation_label(self, annotation_id):
        """获取标注框的标签"""
        if self.current_frame_index in self.annotations:
            for annotation in self.annotations[self.current_frame_index]:
                if annotation['id'] == annotation_id:
                    return annotation['label']
        return ""
    
    # 标注工具组的控制组件按钮点击事件
    def reset_annotation(self):
        """重置当前图片的标注"""
        if self.current_frame_index in self.annotations:
            # 移除当前帧的标注信息
            self.annotations.pop(self.current_frame_index)
            # 清除当前帧的标注组件
            self.annotation_widgets.pop(self.current_frame_index)
            # 刷新标注组件
            self.update_annotation_widgets()
            # 更新当前帧的图片信息
            self.display_current_frame()
        else:
            return

    # 切换到上一帧
    def prev_frame(self):
        """显示上一帧"""
        # 检查是否有视频帧
        if hasattr(self, 'video_frames') and self.video_frames:
            # 检查是否已到达第一帧
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                self.display_current_frame()
                self.update_frame_info()
            else:
                 QMessageBox.information(self, "提示", "已经是第一帧")
        else:
            QMessageBox.information(self, "提示", "请先加载视频")

    # 切换到下一帧
    def next_frame(self):
        """显示下一帧"""
        # 检查是否有视频帧
        if hasattr(self, 'video_frames') and self.video_frames:
            # 检查是否已到达最后一帧
            if self.current_frame_index < self.total_frames - 1:
                self.current_frame_index += 1
                self.display_current_frame()
                self.update_frame_info()
            else:
                QMessageBox.information(self, "提示", "已经是最后一帧")
        else:
            QMessageBox.information(self, "提示", "请先加载视频")

    # 浏览并选择输出TXT路径
    def browse_output_txt_path(self):
        """浏览并选择输出TXT路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择输出TXT文件", self.config['output_txt_path'], "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.config['output_txt_path'] = file_path
            self.output_txt_path_input.setText(file_path)

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

# 主运行函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec())