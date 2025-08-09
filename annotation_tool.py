from hmac import new
import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QFormLayout,
                              QLineEdit, QComboBox, QColorDialog, QTabWidget, QSplitter, QCheckBox, QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer, QEvent
from PySide6.QtGui import QPixmap, QColor, QFont, QImage

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
        # 标注相关变量
        self.annotations = {}          # 存储所有标注信息
        self.annotation_widgets = {}   # 存储所有标注框的文本输入框{frameindex:[{annotationid:widget},{annotationid:widget}]}
        self.current_tool = None       # 当前选中的标注工具
        self.drawing = False           # 是否正在绘制标注
        self.start_point = None        # 绘制开始点
        self.end_point = None          # 绘制结束点
        self.current_color = QColor(Qt.red) # 当前标注的颜色（默认为红色）
        self.current_thickness = 2     # 当前标注的线宽

        # 初始化配置
        self.config = {
            'output_json_path': './output',
            'output_video_path': './processed_videos',
            'background_color': '#f0f0f0',
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
        self.video_display.setMinimumSize(640, 480)      # 初始4：3比例
        self.video_display.setStyleSheet("background-color: #000000; color: white;")
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

        self.rect_tool_btn = QPushButton("矩形")
        self.point_tool_btn = QPushButton("点")

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

        TODO: 添加标注框列表区域和滚动区域
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
        TODO: 连接矩形工具按钮点击事件
        self.rect_tool_btn.clicked.connect(lambda: self.set_current_tool('rectangle'))
        # 连接颜色选择按钮
        self.color_btn.clicked.connect(self.select_color)
        # 连接点工具按钮点击事件
        self.point_tool_btn.clicked.connect(lambda: self.set_current_tool('point'))

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
            self.video_id_value.setText(str(self.video_path.split('/')[-1]))
            self.load_video_frames()
    
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
            # 按帧率提取帧
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.video_frames.append(frame)
            cap.release()
            # 更新界面显示
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_info()
            else:
                QMessageBox.warning(self, "警告", "视频中未提取到帧")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载视频帧时出错: {e}")
            return

    # 显示当前图片帧（⭐⭐⭐⭐⭐）
    def display_current_frame(self):
        """显示当前帧，并绘制标注"""
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index].copy()
            # 转换为QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # 绘制已有的标注
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
            
            # 获取视频显示区域的大小
            display_width = self.video_display.width()
            display_height = self.video_display.height()

            # 计算缩放比例
            scale_factor = min(display_width / width, display_height / height)
            # 保持宽高比
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            scaled_image = q_image.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.video_display.setPixmap(QPixmap.fromImage(scaled_image))

            # 更新标注组件
            self.update_annotation_widgets()

    def update_frame_info(self):
        """更新帧信息显示"""
        self.video_id_value.setText(str(self.video_path.split('/')[-1]).split('.')[0])
        self.total_frame_count = len(self.video_frames)
        self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
        self.fps_value.setText(str(self.frame_rate))


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

    # 设置当前标注工具
    def set_current_tool(self,tool_type):
        """设置当前标注工具"""
        self.current_tool = tool_type
        # 更新按钮样式以显示当前选中的样式
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

    # 事件过滤器
    def eventFilter(self,obj,event):
        """事件过滤器，用于捕获视频显示区域的鼠标事件"""
        if obj is self.video_display and self.current_tool == 'rectangle':
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # 鼠标按下，开始绘制
                self.drawing = True
                self.start_point = event.position().toPoint()
                self.end_point = self.start_point
                return True
            elif event.type() == QEvent.MouseMove and self.drawing:
                # 鼠标移动，更新结束点
                self.end_point = event.position().toPoint()
                # 重绘当前帧
                self.display_current_frame()
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.drawing:
                # 鼠标释放，结束绘制
                self.drawing = False
                self.end_point = event.position().toPoint()
                # 确保起始点和结束点不同
                if self.start_point != self.end_point:
                    self.add_annotation()
                # 重绘当前帧
                self.display_current_frame()
                return True
        return super().eventFilter(obj, event)

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
            'color': (self.current_color.red(), self.current_color.green(), self.current_color.blue())
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
        #TODO: 需要修改根据标注框的id来获取标注框的标签
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