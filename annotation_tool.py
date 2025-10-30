import sys
import os
import cv2
import time
import json
import threading
import copy
from queue import Queue, Empty
import math
import onnxdealA
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QMessageBox, QFrame, QFileDialog, QSlider, QGroupBox, QFormLayout,
                              QLineEdit, QComboBox, QColorDialog, QTabWidget, QSplitter, QCheckBox, QSizePolicy,QStyle,QInputDialog)
from PySide6.QtCore import Qt, QTimer, QEvent, QPoint, QRect,QSize
from PySide6.QtGui import QFont, QPixmap, QCursor, QColor, QImage

# 分割标注工具（目前可能不需要）
class SegmentationAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("分割标注")
        self.resize(1200, 800)
        
        # 配置参数
        self.config = {
            'background_color': '#f5f5f5',
            'default_input_path': 'input_videos/',
            'output_path': 'output/',
            'model_path': 'model/yolov8m_gray.onnx',
            'classes_path': 'model/classes.txt'
        }
        
        # 分割标注特有变量
        self.segmentation_masks = {}
        self.current_brush_size = 5
        self.brush_color = QColor(Qt.red)
        self.current_tool = 'brush'
        
        # 基础变量
        self.video_path = ""
        self.current_frame_index = 0
        self.total_frame_count = 0
        self.frame_rate = 30
        self.video_frames = []
        
        # 创建中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 初始化UI
        self.init_ui()
    
    def update_brush_size(self, size):
        self.current_brush_size = size
        self.statusBar().showMessage(f"画笔大小: {size}")
    
    def select_brush_color(self):
        color = QColorDialog.getColor(self.brush_color, self, "选择画笔颜色")
        if color.isValid():
            self.brush_color = color
            self.color_btn.setStyleSheet(f'background-color: {color.name()};')
    
    def set_current_tool(self, tool_type):
        self.current_tool = tool_type
        self.brush_btn.setChecked(tool_type == 'brush')
        self.eraser_btn.setChecked(tool_type == 'eraser')
    
    def browse_video_directory(self):
        """打开文件选择对话框，支持视频、图片文件和图片文件夹"""
        # 先显示文件选择对话框
        default_path = self.config['default_input_path']
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图片文件", default_path,
            "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.load_video_frames()
        else:
            # 如果用户取消了文件选择，询问是否要选择文件夹
            reply = QMessageBox.question(
                self, "选择文件夹", "是否要选择包含图片的文件夹?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                directory_path = QFileDialog.getExistingDirectory(
                    self, "选择图片文件夹", default_path
                )
                if directory_path:
                    self.video_path = directory_path
                    self.load_image_folder()
                    
    def load_image_folder(self):
        """批量加载文件夹中的图片"""
        try:
            # 清空之前的帧数据
            self.video_frames = []
            self.current_frame_index = 0
            self.total_frame_count = 0
            self.segmentation_masks = {}
            
            # 获取文件夹中的所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            files = [f for f in os.listdir(self.video_path) if os.path.splitext(f)[1].lower() in image_extensions]
            
            # 按文件名排序，确保图片按顺序加载
            files.sort()
            
            if not files:
                QMessageBox.warning(self, "警告", "所选文件夹中没有找到图片文件")
                return
            
            # 加载所有图片
            for file in files:
                file_path = os.path.join(self.video_path, file)
                frame = cv2.imread(file_path)
                if frame is not None:
                    # 转换为RGB格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_frames.append(frame)
                else:
                    print(f"无法加载图片: {file}")
            
            # 更新视频信息
            self.total_frame_count = len(self.video_frames)
            self.frame_rate = 1  # 图片序列的帧率设为1
            
            # 初始化当前帧索引
            self.current_frame_index = 0
            
            # 显示当前帧
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_label()
                QMessageBox.information(self, "加载成功", f"成功加载 {self.total_frame_count} 张图片")
            else:
                QMessageBox.warning(self, "警告", "未加载到任何图片")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载文件夹时出错: {str(e)}")
            print(e)
    
    def load_video_frames(self):
        """加载视频或图片文件并处理为帧"""
        try:
            self.video_frames = []
            
            # 检查文件类型
            file_ext = os.path.splitext(self.video_path)[1].lower()
            
            if file_ext in ['.mp4', '.avi', '.mov']:
                # 加载视频文件
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    QMessageBox.warning(self, "警告", "无法打开视频文件")
                    return
                
                # 获取视频信息
                self.total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                
                # 读取所有帧
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 转换为RGB格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_frames.append(frame)
                
                cap.release()
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 加载单个图片文件
                frame = cv2.imread(self.video_path)
                if frame is None:
                    QMessageBox.warning(self, "警告", "无法打开图片文件")
                    return
                
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_frames.append(frame)
                self.total_frame_count = 1
                self.frame_rate = 1
                
            # 初始化当前帧索引
            self.current_frame_index = 0
            
            # 显示当前帧
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_label()
                QMessageBox.information(self, "加载成功", f"成功加载 {self.total_frame_count} 帧")
            else:
                QMessageBox.warning(self, "警告", "未加载到任何帧")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载文件时出错: {str(e)}")
            print(e)
    
    def display_current_frame(self):
        """显示当前帧，并绘制分割掩码"""
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index].copy()
            
            # 绘制分割掩码
            if self.current_frame_index in self.segmentation_masks:
                mask = self.segmentation_masks[self.current_frame_index]
                # 这里可以添加掩码绘制逻辑
                # 例如: frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
                pass
            
            # 转换为QImage并显示
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 调整显示大小
            scaled_pixmap = QPixmap.fromImage(qimage).scaled(
                self.video_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_frame.setPixmap(scaled_pixmap)
    
    def update_frame_label(self):
        """更新帧信息标签"""
        if self.total_frame_count > 0:
            self.frame_label.setText(f'帧: {self.current_frame_index + 1}/{self.total_frame_count}')
        else:
            self.frame_label.setText('帧: 0/0')
    
    def prev_frame(self):
        """显示上一帧"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.display_current_frame()
            self.update_frame_label()
    
    def next_frame(self):
        if self.current_frame_index < self.total_frame_count - 1:
            self.current_frame_index += 1
            self.display_current_frame()
            self.update_frame_label()
    
    def save_segmentation(self):
        # 实现分割结果保存逻辑
        QMessageBox.information(self, "保存成功", "分割标注已保存")
    
    def export_segmentation(self):
        """导出分割结果"""
        # 实现分割结果导出逻辑
        QMessageBox.information(self, "导出成功", "分割结果已导出")
    
    def undo_segmentation(self):
        """撤销上一步操作"""
        self.statusBar().showMessage("撤销操作")
        
    def set_style(self):
        """设置分割标注工具样式"""
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QPushButton { padding: 8px 16px; border-radius: 4px; }
            QLabel { font-size: 14px; }
            QSlider::groove:horizontal { height: 6px; background: #ddd; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; height: 16px; background: #4a90e2; border-radius: 8px; }
        """)
    
    def create_menu_bar(self):
        """创建分割标注工具菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        open_action = file_menu.addAction('打开视频')
        open_action.triggered.connect(self.browse_video_directory)
        
        save_action = file_menu.addAction('保存标注')
        save_action.triggered.connect(self.save_segmentation)
        
        export_action = file_menu.addAction('导出分割结果')
        export_action.triggered.connect(self.export_segmentation)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑')
        undo_action = edit_menu.addAction('撤销')
        undo_action.triggered.connect(self.undo_segmentation)
    
    def create_tool_bar(self):
        """创建分割标注工具工具栏"""
        toolbar = self.addToolBar('分割工具')
        toolbar.setMovable(False)
        
        # 画笔大小控制
        size_label = QLabel('画笔大小:')
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 20)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        
        # 颜色选择
        self.color_btn = QPushButton('颜色')
        self.color_btn.setStyleSheet(f'background-color: {self.brush_color.name()};')
        self.color_btn.clicked.connect(self.select_brush_color)
        
        # 工具按钮
        self.brush_btn = QPushButton('画笔')
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(lambda: self.set_current_tool('brush'))
        
        self.eraser_btn = QPushButton('橡皮擦')
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self.set_current_tool('eraser'))
        
        # 添加到工具栏
        toolbar.addWidget(size_label)
        toolbar.addWidget(self.brush_size_slider)
        toolbar.addWidget(self.color_btn)
        toolbar.addSeparator()
        toolbar.addWidget(self.brush_btn)
        toolbar.addWidget(self.eraser_btn)
    
    def create_segmentation_content(self):
        """创建分割标注主内容区域"""
        # 创建分割面板
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧视频显示区域
        self.video_frame = QLabel('视频显示区域')
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setStyleSheet('background-color: black; color: white;')
        self.video_frame.setMinimumSize(800, 600)
        
        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 帧控制
        frame_control = QHBoxLayout()
        self.prev_btn = QPushButton('上一帧')
        self.prev_btn.clicked.connect(self.prev_frame)
        
        self.next_btn = QPushButton('下一帧')
        self.next_btn.clicked.connect(self.next_frame)
        
        self.frame_label = QLabel('帧: 0/0')
        
        frame_control.addWidget(self.prev_btn)
        frame_control.addWidget(self.next_btn)
        frame_control.addWidget(self.frame_label)
        
        # 分割标签
        label_group = QGroupBox('分割标签')
        label_layout = QVBoxLayout()
        self.label_combo = QComboBox()
        self.label_combo.addItems(['背景', '猪', '饲料', '其他'])
        label_layout.addWidget(self.label_combo)
        label_group.setLayout(label_layout)
        
        # 添加到控制布局
        control_layout.addLayout(frame_control)
        control_layout.addWidget(label_group)
        control_layout.addStretch()
        
        # 添加到分割器
        splitter.addWidget(self.video_frame)
        splitter.addWidget(control_panel)
        splitter.setSizes([800, 300])
        
        self.main_layout.addWidget(splitter)
    
    def init_ui(self):
        # 分割标注界面初始化 - 基于方框标注工具修改
        self.video_frames = []
        self.current_frame_index = 0
        self.segmentation_masks = {}
        self.current_tool = 'brush'
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 设置样式
        self.set_style()
        
        # 创建菜单栏和工具栏
        self.create_menu_bar()
        self.create_tool_bar()
        
        # 创建主内容区域（分割标注特有的布局）
        self.create_segmentation_content()
        
        # 状态栏
        self.statusBar().showMessage("分割标注工具就绪")

# 入口选择界面
class SelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("功能选择")
        self.resize(600, 400)
        
        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 设置样式
        self.set_style()
        
        # 创建标题
        self.create_title()
        
        # 创建功能选择按钮
        self.create_selection_buttons()
        
        # 创建底部信息
        self.create_bottom_info()
        
        # 居中窗口
        self.center_on_screen()
    
    def set_style(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QPushButton { 
                padding: 12px 24px; 
                font-size: 16px; 
                border-radius: 8px; 
                border: 2px solid #4a90e2;
                background-color: white;
                color: #4a90e2;
            }
            QPushButton:hover { 
                background-color: #4a90e2;
                color: white;
            }
            QFrame { 
                background-color: white;
                border-radius: 10px;
            }
        """)
    
    def create_title(self):
        """创建标题部分"""
        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel("功能选择")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("请选择您要使用的功能")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        self.main_layout.addWidget(title_frame)
        self.main_layout.addSpacing(30)
    
    def create_selection_buttons(self):
        """创建功能选择按钮"""
        buttons_frame = QFrame()
        buttons_layout = QVBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(40, 20, 40, 20)
        buttons_layout.setSpacing(15)
        
        # 方框标注工具按钮
        self.box_annotation_btn = QPushButton("方框标注")
        self.box_annotation_btn.clicked.connect(self.open_box_annotation)
        buttons_layout.addWidget(self.box_annotation_btn)
        
        # 分割标注工具按钮
        self.segmentation_annotation_btn = QPushButton("分割标注")
        self.segmentation_annotation_btn.setStyleSheet("""
            QPushButton:disabled {
            background-color: #f0f0f0;
            color: #808080;
            padding: 12px 24px; 
            font-size: 16px; 
            border-radius: 8px; 
            border: 2px solid #d0d0d0;
            }
        """)
        self.segmentation_annotation_btn.setEnabled(False)
        self.segmentation_annotation_btn.clicked.connect(self.open_segmentation_annotation)
        buttons_layout.addWidget(self.segmentation_annotation_btn)
        
        self.main_layout.addWidget(buttons_frame)
    
    def create_bottom_info(self):
        """创建底部信息"""
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(20, 10, 20, 10)
        
        version_label = QLabel("版本 1.6.0")
        version_label.setAlignment(Qt.AlignCenter)
        
        bottom_layout.addWidget(version_label)
        self.main_layout.addWidget(bottom_frame)
        self.main_layout.setAlignment(bottom_frame, Qt.AlignBottom)
    
    def center_on_screen(self):
        """将窗口居中显示在屏幕上"""
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def open_box_annotation(self):
        try:
            # 隐藏当前选择窗口
            self.hide()
            # 创建并显示方框标注工具窗口
            self.box_annotation_window = BoxAnnotationTool()
            original_close_event = self.box_annotation_window.closeEvent
            # 重写closeEvent方法，确保先调用原始方法再显示选择窗口
            def new_close_event(event):
                # 调用原始的closeEvent方法进行资源清理
                original_close_event(event)
                # 显示选择窗口
                self.show_after_close(event)
            # 连接关闭信号
            self.box_annotation_window.closeEvent = new_close_event
            self.box_annotation_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开方框标注工具时出错: {str(e)}")
            self.show()
    
    def open_segmentation_annotation(self):
        '''
        try:
            # 隐藏当前选择窗口
            self.hide()
            # 创建并显示分割标注工具窗口
            self.segmentation_annotation_window = SegmentationAnnotationTool()
            self.segmentation_annotation_window.show()
            # 连接关闭信号
            self.segmentation_annotation_window.closeEvent = lambda event: self.show_after_close(event)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开分割标注工具时出错: {str(e)}")
            self.show()
        '''
        pass

    def show_after_close(self, event):
        """当子窗口关闭后，显示选择窗口"""
        self.show()
        event.accept()

# 方框标注工具
class BoxAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("方框标注")
        self.resize(1200, 800)
        # 添加线程锁，用于解决next_frame连点问题
        self.next_frame_lock = threading.Lock()
        self.loading = False           # 是否正在加载视频
        # 视频帧相关变量
        self.video_frames = []         # 存储所有视频帧
        self.video_path = ""           # 视频的输入路径
        self.video_type = False        # 判断是视频还是图片序列
        self.current_frame_index = 0   # 当前显示的帧索引
        self.total_frame_count = 0     # 视频总帧数
        self.frame_rate = 30           # 视频的默认帧率是30
        self.video_frame_selection_interval = 4 # 视频帧跳跃间隔（默认取第1帧、第5帧、第9帧...）
        self.key_frames = {}           # 存储视频的关键帧索引
        self.frame_queue = Queue(maxsize=150)    # 图片帧读取队列
        # 标注相关变量
        self.original_annotations = {}  # 存储原始标注信息（作为备份）
        self.annotations = {}          # 存储所有标注信息
        self.current_tool = None       # 当前选中的标注工具
        # 标注框绘制相关变量
        self.drawing = False           # 是否正在绘制标注
        self.start_point = None        # 绘制开始点
        self.end_point = None          # 绘制结束点
        self.current_color = QColor(Qt.green)    # 当前手动标注的颜色（默认为绿色）
        self.current_thickness = 2     # 当前标注的线宽
        # 标注框拖动相关变量
        self.dragging = False          # 是否正在拖动标注
        self.dragging_annotation = None # 当前正在拖动的标注
        self.drag_offset = QPoint()    # 拖动偏移量
        # 文件保存相关变量
        self.saving_timestamp = None   # 记录当前项目第一次保存的时间戳
        # 标注框扩缩相关变量
        self.resizing = False          # 是否正在调整标注框大小
        self.resize_anchor = None      # 调整大小的锚点位置('n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw')
        self.resizing_annotation = None   # 当前正在调整大小的标注
        # 标签和高亮相关变量
        self.selected_annotation = None   # 当前帧被选定的标注框
        self.category_labels = {}      # 存储右侧窗口中的类别标签组件

        # 初始化配置
        self.config = {
            'default_input_path': './input_atlas',
            'output_txt_path': './output',
            'output_video_path': './processed_videos',
            'background_color': '#f0f0f0',
            'model_path': './model/pig_gesture_best.onnx',
            'classes_path': './model/classes.txt'
        }

        # 初始化类别字典
        self.classes = self.init_classes()

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
        load_video_action = file_menu.addAction("加载文件")
        load_video_action.triggered.connect(self.load_video)

        save_project_action = file_menu.addAction("保存项目")
        save_project_action.triggered.connect(self.save_project)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助")
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)

    def create_tool_bar(self):
        """创建工具栏"""
        tool_bar = self.addToolBar("工具")
        tool_bar.setMovable(False)

        load_video_btn = QPushButton("加载文件")
        load_video_btn.setStyleSheet("QPushButton { padding: 4px 8px; border: none;}")
        load_video_btn.clicked.connect(self.load_video)
        tool_bar.addWidget(load_video_btn) 

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

         # 添加体重输入框
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("输入体重...")
        self.weight_input.setFixedWidth(100)
        self.weight_input.textChanged.connect(self.update_frame_weight)

        essential_frame_layout.addWidget(self.essential_frame_label)
        essential_frame_layout.addWidget(self.essential_frame_checkbox)
        essential_frame_layout.addWidget(self.weight_input)  # 添加输入框到勾选框右侧
        essential_frame_layout.addStretch()
        info_layout.addLayout(top_info_layout)
        info_layout.addLayout(essential_frame_layout)
        left_layout.addWidget(info_group)

        # 视频显示区域
        #TODO: 视频显示区域在界面可缩放时保持原视频比例
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

        tool_layout.addWidget(self.mouse_btn)
        tool_layout.addWidget(self.rect_tool_btn)

        tool_group.setLayout(tool_layout)
        annotation_layout.addWidget(tool_group)

        # 标注属性组
        props_group = QGroupBox("标注属性")
        props_layout = QFormLayout()

        self.color_btn = QPushButton("选择颜色")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 10)
        self.thickness_slider.setValue(2)

        props_layout.addRow("颜色:", self.color_btn)
        props_layout.addRow("线宽:", self.thickness_slider)

        props_group.setLayout(props_layout)
        annotation_layout.addWidget(props_group)

        # 添加识别到的类型信息
        self.current_category_group = QGroupBox("当前类别")
        self.category_layout = QVBoxLayout()      # 创建一个垂直布局管理器
        self.current_category_group.setLayout(self.category_layout)     # 定义了该分组框内控件的布局规则
        annotation_layout.addWidget(self.current_category_group)        
        
        #添加重置该标签、上一帧、下一帧的按钮组
        control_group = QGroupBox("控制")
        # 创建一个垂直布局作为主布局
        main_control_layout = QVBoxLayout()

        # 第一行布局：重置按钮和上下帧按钮
        first_row_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重置该标签")
        self.prev_frame_btn = QPushButton("上一帧")
        self.next_frame_btn = QPushButton("下一帧")
        
        # 添加k值输入框和前k帧、后k帧按钮
        self.k_value_input = QLineEdit("5")  # 默认值为5
        self.k_value_input.setFixedWidth(50)
        self.prev_k_frame_btn = QPushButton("前k帧")
        self.next_k_frame_btn = QPushButton("后k帧")

        # 连接按钮点击事件
        self.rect_tool_btn.clicked.connect(lambda: self.set_current_tool('rectangle'))
        # 连接颜色选择按钮
        self.color_btn.clicked.connect(self.select_color)
        # 连接鼠标拖动按钮点击事件
        self.mouse_btn.clicked.connect(lambda: self.set_current_tool('mouse'), self.mouse_drag)

        self.reset_btn.clicked.connect(self.reset_annotation)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.prev_k_frame_btn.clicked.connect(self.prev_k_frames)
        self.next_k_frame_btn.clicked.connect(self.next_k_frames)

        # 将按钮添加到第一行布局
        first_row_layout.addWidget(self.reset_btn)
        first_row_layout.addWidget(self.prev_frame_btn)
        first_row_layout.addWidget(self.next_frame_btn)

        # 创建第二行布局：k值相关组件
        second_row_layout = QHBoxLayout()
        # 将k值控制布局添加到第二行布局
        k_label = QLabel("k值:")
        k_label.setAlignment(Qt.AlignCenter)
        # 设置加粗
        k_label.setStyleSheet("font-weight: bold;")
        second_row_layout.addWidget(k_label)
        second_row_layout.addWidget(self.k_value_input)
        second_row_layout.addWidget(self.prev_k_frame_btn)
        second_row_layout.addWidget(self.next_k_frame_btn)
        
        # 将两行布局添加到主布局
        main_control_layout.addLayout(first_row_layout)
        main_control_layout.addLayout(second_row_layout)
    
        control_group.setLayout(main_control_layout)
        annotation_layout.addWidget(control_group)

        annotation_layout.addStretch()
        right_panel.addTab(annotation_tab, "标注")

        # 输出设置标签页
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)

        output_group = QGroupBox("输入输出设置")
        output_form_layout = QFormLayout()

        self.video_path_input = QLineEdit(self.config['default_input_path'])
        self.video_path_input.setReadOnly(True)
        self.browse_input_btn = QPushButton("更改输入图片集路径")
        self.browse_input_btn.setStyleSheet("border : 1px solid black;")
        self.browse_input_btn.clicked.connect(self.browse_input_video_path)

        self.output_txt_path_input = QLineEdit(self.config['output_txt_path'])
        self.output_txt_path_input.setReadOnly(True)
        self.browse_output_btn = QPushButton("更改输出参数文件路径")
        self.browse_output_btn.setStyleSheet("border : 1px solid black;")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)

        self.output_video_path_input = QLineEdit(self.config['output_video_path'])
        self.output_video_path_input.setReadOnly(True)
        self.browse_output_video_btn = QPushButton("更改输出标注视频帧路径")
        self.browse_output_video_btn.clicked.connect(self.browse_video_directory)
        
        output_form_layout.addRow("输入图片集路径:", self.video_path_input)
        output_form_layout.addRow("", self.browse_input_btn)
        output_form_layout.addRow("输出参数文件路径:", self.output_txt_path_input)
        output_form_layout.addRow("", self.browse_output_btn)
        output_form_layout.addRow("输出标注视频图像帧路径:", self.output_video_path_input)
        output_form_layout.addRow("", self.browse_output_video_btn)

        output_group.setLayout(output_form_layout)
        output_layout.addWidget(output_group)

        output_layout.addStretch()
        right_panel.addTab(output_tab, "输入与输出")

        # 设置分割器初始大小
        main_splitter.setSizes([800, 400])
    
    def init_classes(self):
        """ 初始化类别字典 """
        classes = {}
        with open(self.config['classes_path'], 'r') as f:
            for i,line in enumerate(f):
                line = line.strip()
                if line:
                    cls_name = line.split()[0]
                    classes[i] = cls_name
        return classes

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 确保视频显示区域居中
        self.video_display.parent().layout().setAlignment(self.video_display, Qt.AlignCenter)

    # 从本地文件夹加载视频
    def load_video(self):
        """加载视频文件"""
        # 在加载新视频前清空当前视频信息和变量
        self.clear_video_resources()
        # 先判断默认输入文件夹是否是图片数据集
        default_input_path = self.config['default_input_path']
        if os.path.exists(default_input_path):
            files = os.listdir(default_input_path)
            # 文件必须存在且全部都是图片文件才能加载
            if files and all(file.endswith(('.jpg', '.jpeg', '.png')) for file in files):
                # 全部是图片文件，直接加载
                self.video_path = default_input_path
                self.video_id_value.setText(str(self.video_path.split('/')[-1]))
                self.load_default_atlas(files)
                return
        # 如果默认输入文件夹没有图片文件，从本地文件夹加载视频
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或图片文件", "", "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self.video_path = file_path
            self.video_id_value.setText(str(self.video_path.split('/')[-1]))
            self.load_video_frames()
            self.video_type = True
    
    # 提取模型识别的自动标注信息
    def Extract_the_annotation_information(self,frame):
        """ 提取单帧模型识别的自动标注信息 """
        # 返回一个列表，每个列表包含自动标注框的信息
        # 返回的格式：[{'cls': 0, 'xyxy': [x1,y1,x2,y2], 'score': 0.8},{'cls': 1, 'xyxy': [x1,y1,x2,y2], 'score': 0.8}]
        if frame is None:
            return None
        result = onnxdealA.main(self.config['model_path'],frame,self.config['classes_path'])
        return result
    
    # 根据识别到的标注信息进行保存
    def Save_model_recognition_annotations(self,result,frame_index):
        """ 根据识别到的标注信息进行保存 """
        if not result:
            return
        # 处理检测结果
        frame_annotation = []
        for i,box in enumerate(result):
            # 获取标注框的坐标，将列表结构拆分成四个变量
            x1,y1,x2,y2 = box['xyxy']
            # 获取置信度
            # confidence = float(box.conf[0])
            # 获取检测类别
            class_id = box['cls']
            # 构建标注信息
            annotation ={
                'id': i + 1,
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                # 'confidence': confidence,
                'label': f"pig{i+1}",
                'text': '',
                'color': (0, 255, 0),  # 自动化标注框默认是绿色
                # 记录原有颜色
                'original_color': (0, 255, 0)  # 自动化标注框默认是绿色
            }
            frame_annotation.append(annotation)
        # 保存标注信息
        if frame_annotation:
            # 将标注信息加入原始标注字典（作为备份），使用深拷贝创建独立副本
            self.original_annotations[frame_index] = copy.deepcopy(frame_annotation)
            # 初始化当前标注字典，使用深拷贝创建独立副本
            self.annotations[frame_index] = copy.deepcopy(frame_annotation)

    # 加载图片数据集功能
    def load_default_atlas(self, files, montage=True):
        """批量加载文件夹中的图片"""
        try:
            # 清空之前的帧数据
            self.video_frames = []
            self.current_frame_index = 0
            self.annotations = {}
            
            # 加载所有图片
            for file in files:
                if not montage:
                    file_path = self.video_path
                else:
                    file_path = os.path.join(self.video_path, file)
                frame = cv2.imread(file_path)
                if frame is not None:
                    # 转换为RGB格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_frames.append(frame)
                else:
                    print(f"无法加载图片: {file}")
            
            # 更新视频信息
            self.total_frame_count = len(self.video_frames)
            self.frame_rate = 1  # 图片序列的帧率设为1

            # 判断是否有指定的模型
            if self.config['model_path'] is not None:
                for frame_index,frame in enumerate(self.video_frames):
                    result = self.Extract_the_annotation_information(frame)
                    self.Save_model_recognition_annotations(result, frame_index)
            
            # 更新界面显示
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_info()
                QMessageBox.information(self, "加载完成", f"成功加载 {len(self.video_frames)} 张图片")
            else:
                QMessageBox.warning(self, "警告", "未加载到任何图片")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载图片时出错: {e}")
            print(e)

    # 将视频切换成图片帧
    def load_video_frames(self):
        """按帧率从视频中提取帧"""
        # 清空之前的帧数据
        self.video_frames = []
        self.current_frame_index = 0
        self.original_annotations = {}
        self.annotations = {}
        try:
            # 图像格式
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            # 查看self.video_path路径最后一个文件的后缀
            file_name = str(self.video_path.split('/')[-1])
            _, file_extension = os.path.splitext(file_name)
            # 如果文件是图像类型的
            if file_extension in image_extensions:
                self.load_default_atlas([file_name], False)
                return

            # 打开视频文件
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开视频文件")
                return
            # 读取视频信息，直接从视频文件的元数据获取信息，这是一个高效信息
            self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
            # 处理总帧数等于视频总帧数除以间隔并向上取整
            # self.total_frame_count = int(math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT)/self.video_frame_selection_interval))

            # 图片帧读取线程    
            def frame_reader(cap:cv2.VideoCapture):
                frame_count = 0
                self.loading = True
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 从第一张开始，每隔k张读取一帧
                    if frame_count % self.video_frame_selection_interval == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 解析当前帧的标注信息
                        if self.config['model_path'] is not None:
                            result = self.Extract_the_annotation_information(frame)
                            # 保存当前帧的标注信息
                            self.Save_model_recognition_annotations(result, int((frame_count / self.video_frame_selection_interval)))
                        self.frame_queue.put(frame)
                        self.total_frame_count += 1
                    frame_count += 1
                    # 实时更新界面显示
                    self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count} (加载中...)")
                # 视频帧读取完毕后，放入None作为结束信号
                self.frame_queue.put(None)
                self.loading = False
                # 通知主线程更新界面（加载完成）
                self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
                cap.release()

            # 1. 先将视频帧转换为图片帧，启动视频帧读取子线程，当视频读取结束后，视频帧读取子线程会结束
            self.reader_thread = threading.Thread(target=frame_reader, args=(self.cap,))
            self.reader_thread.daemon = True
            self.reader_thread.start()

            # 2. 从视频帧队列中提取帧
            # 2.1 先提取第一帧
            frame = self.frame_queue.get()
            self.video_frames.append(frame)
            
            # 更新第一帧界面显示
            if self.video_frames:
                self.display_current_frame()
                self.update_frame_info()
                # 如果有模型检测结果，显示提示
                if self.annotations:
                    # detected_count = sum(len(anns) for anns in self.annotations.values())
                    # QMessageBox.information(self, "检测结果", f"检测到 {detected_count} 只猪")
                    # 输出检测完成
                    QMessageBox.information(self, "检测成功", "视频检测已完成，结果已暂存。")
            else:
                QMessageBox.warning(self, "警告", "视频中未提取到帧")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载视频帧时出错: {e}")
            print(e)
            return

    # 显示当前图片帧（⭐⭐⭐⭐⭐）
    def display_current_frame(self):
        """显示当前帧，并绘制标注"""
        # 可以直接从已经被yolov8处理过的帧提取结果
        if 0 <= self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index].copy()
            # 转换为QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # 绘制已有的标注（包含yolov8识别结果）,已包含拖动的标注框
            if self.current_frame_index in self.annotations:
                for annotation in self.annotations[self.current_frame_index]:
                    # 绘制矩形
                    color = annotation['color']
                    thickness = 2
                    # 如果是当前被选中的标注框，高亮显示
                    if annotation == self.selected_annotation:
                        thickness = 3  # 红色
                    cv2.rectangle(frame, (annotation['x1'], annotation['y1']), 
                                  (annotation['x2'], annotation['y2']), color, thickness)
                    '''
                    # 绘制标签
                    label_text = f"{annotation['id']}"
                    cv2.putText(frame, label_text, (annotation['x1'], annotation['y1'] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    '''
                    # 绘制标签，包含标签和类别信息
                    class_name = self.classes.get(annotation['class_id'], annotation['class_id'])
                    # 为了避免OpenCV的中文显示问题，使用英文格式
                    label_text = f"pig{annotation['id']}: {class_name}"
                    # 增大字体大小到0.8，调整位置在方框左上角
                    font_scale = 0.8
                    font_thickness = 2
                    # 添加标签背景以提高可读性
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    text_x = annotation['x1']
                    text_y = annotation['y1'] - 5 if annotation['y1'] > text_height + 10 else annotation['y1'] + text_height + 15
                    # 绘制背景矩形
                    cv2.rectangle(frame, (text_x - 3, text_y - text_height - baseline), 
                                  (text_x + text_width + 3, text_y + 3), (255, 255, 255), -1)
                    # 绘制边框
                    cv2.rectangle(frame, (text_x - 3, text_y - text_height - baseline), 
                                  (text_x + text_width + 3, text_y + 3), color, 1)
                    # 绘制文本 - OpenCV默认字体不支持中文，所以使用英文
                    cv2.putText(frame, label_text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                    
            # 绘制正在绘制的矩形
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(frame, (self.start_point.x(), self.start_point.y()),
                              (self.end_point.x(), self.end_point.y()), 
                              (self.current_color.red(), self.current_color.green(), self.current_color.blue()), 2)
            
            # 获取视频显示区域的大小
            display_width = self.video_display.width()
            display_height = self.video_display.height()

            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            scaled_image = q_image.scaled(display_width,display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.video_display.setPixmap(QPixmap.fromImage(scaled_image))

    # 更新当前帧的图片信息（视频名称、总帧数、当前帧数、是否为关键帧、fps、当前标注框的类别和标签）
    def update_frame_info(self):
        """更新帧信息显示"""
        self.video_id_value.setText(str(self.video_path.split('/')[-1]).split('.')[0])
        if self.loading:
            self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count} (加载中...)")
        else:
            self.frame_num_value.setText(f"{self.current_frame_index + 1}/{self.total_frame_count}")
        self.fps_value.setText(str(self.frame_rate))
        self.essential_frame_checkbox.setChecked(self.key_frames.get(self.current_frame_index, False))
        if  self.current_frame_index in self.annotations and self.annotations[self.current_frame_index]:
            self.weight_input.setText(self.annotations[self.current_frame_index][0].get('text', ''))
        else:
            self.weight_input.setText('')
        # 更新当前图片的类别信息列表
        self.update_category_list()
    
    # 更新类别列表
    def update_category_list(self):
        # 更新当前图片的类别信息列表
        # 先将之前的类别标签清除
        while self.category_layout.count() > 0:
            item = self.category_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.category_labels.clear()
        if self.current_frame_index in self.annotations:
            # 显示当前帧的所有类别和标签
            for annotation in self.annotations[self.current_frame_index]:
                # 获取类别名称，如果没有定义则显示无类别
                class_name = self.classes.get(annotation['class_id'], '无类别')
                # 创建显示标注标签和类别的文本
                label_text = f"{annotation['label']}: {class_name}"

                # 创建水平布局容器
                h_layout = QHBoxLayout()
                h_layout.setSpacing(5)
                h_layout.setContentsMargins(2, 2, 2, 2)
                
                # 创建标签控件
                category_label = QLabel(label_text)
                # 设置默认样式，增大字体并确保中文正常显示
                category_label.setStyleSheet("font-size: 14px; color: #666; background-color: transparent; font-family: SimHei, Microsoft YaHei, sans-serif;")
                # 设置鼠标指针为手型
                category_label.setCursor(Qt.PointingHandCursor)
                # 添加点击事件
                category_label.mousePressEvent = lambda event, annotation=annotation: self.highlight_annotation(annotation)

                # 创建删除按钮（叉号）
                delete_button = QPushButton("X")
                delete_button.setFixedSize(20, 20)
                delete_button.setStyleSheet("font-size: 16px; color: red; background-color: transparent; border: 1px solid #ccc; border-radius: 2px;")
                delete_button.setCursor(Qt.PointingHandCursor)
                # 连接删除按钮的点击事件，删除当前类别标签
                delete_button.clicked.connect(lambda checked,anno=annotation: self.delete_annotation(anno))

                # 将标签和删除按钮添加到水平布局
                h_layout.addWidget(category_label)
                h_layout.addWidget(delete_button)

                # 创建一个容器小部件来容纳水平布局
                container_widget = QWidget()
                container_widget.setLayout(h_layout)

                # 存储容器控件引用，以标注ID为键
                self.category_labels[annotation['id']] = container_widget

                # 将容器小部件添加到类别布局中
                self.category_layout.addWidget(container_widget)
        # 更新类别标签高亮状态
        self.update_category_labels_highlight(self.selected_annotation)
        # 如果焦点在体重输入框上，消除输入框的焦点
        if self.weight_input.hasFocus():
            self.weight_input.clearFocus()

    # 保存项目标注信息到txt和json文件
    def save_project(self):
        """保存标注信息到txt和json文件,每个帧一个文件,txt和json分开保存"""
        # 获取视频名称作为文件夹名
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        # 检查当前视频名称是否为空
        if not video_name:
            QMessageBox.warning(self, "视频名称为空", "请先加载视频文件。")
            return
        # 确保输出目录存在，使用视频名称加时间戳作为子文件夹
        if self.saving_timestamp is None:
            self.saving_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        # 创建输出目录下的二级目录
        file_name = video_name + '_' + self.saving_timestamp
        
        # 创建两个不同的目录，分别用于保存txt和json文件
        txt_output_dir = os.path.join(self.config['output_txt_path'], file_name, 'txt')
        json_output_dir = os.path.join(self.config['output_txt_path'], file_name, 'json')
        
        # 创建目录
        if not os.path.exists(txt_output_dir):
            os.makedirs(txt_output_dir)
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        # 准备保存的数据
        project_data = {
            # 基本配置信息，实际不输出，先保留
            'video_path': self.video_path,
            'frame_rate': self.frame_rate,
            'total_frames': self.total_frame_count,
            # 图片帧标注信息
            'annotations': self.annotations
        }

        # 保存到TXT和JSON文件，每个帧一个文件
        try:
            for frame_idx, annotations in project_data['annotations'].items():
                # 将frame_idx从浮点数转换为整数
                frame_idx_int = int(frame_idx)
                # 第一步检查，判断当前帧是否为关键帧，如果不是则跳过不输出
                if not self.key_frames.get(frame_idx_int, False):
                    continue

                # 为每个帧创建单独的TXT文件，文件名格式为frame_xxxxx.txt
                txt_file_name = f"frame_{frame_idx_int:05d}.txt"
                txt_file_path = os.path.join(txt_output_dir, txt_file_name)
                
                # 为每个帧创建单独的JSON文件，文件名格式为frame_xxxxx.json
                json_file_name = f"frame_{frame_idx_int:05d}.json"
                json_file_path = os.path.join(json_output_dir, json_file_name)
                
                # 存储单帧JSON数据
                json_single_frame_data = []
                
                # 输出txt文件
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    # 处理当前帧的所有标注信息
                    for ann in annotations:
                        x_min, y_min = ann['x1'], ann['y1']
                        x_max, y_max = ann['x2'], ann['y2']
                        class_id = ann['class_id']
                        text = ann['text']
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
                        # 写入JSON数据中
                        json_single_frame_data.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'text': text,
                        })
                        f.write(f"{class_id},{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f},{text}\n")
                
                # 写入单帧JSON文件
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_single_frame_data, f, ensure_ascii=False, indent=4)
                
                # 清空单帧数据
                json_single_frame_data = []
                
            # 显示成功消息
            QMessageBox.information(self, "成功", f"所有关键帧已保存\nTXT文件目录: {txt_output_dir}\nJSON文件目录: {json_output_dir}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存项目时出错: {e}")

    # 显示关于对话框（目前直接pass）
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
        # 如果annotations中不存在当前帧的标注，创建一个空列表
        if self.current_frame_index not in self.annotations:
            self.annotations[self.current_frame_index] = []

    # 更新当前帧的体重信息
    def update_frame_weight(self, text):
        """更新当前帧的体重信息"""
        # 如果文本框非空，自动将当前帧标记为关键帧
        if text.strip() and not self.key_frames.get(self.current_frame_index, False):
            self.key_frames[self.current_frame_index] = True
            self.essential_frame_checkbox.setChecked(True)
        # 如果文本框为空，自动将当前帧标记为非关键帧
        elif not text.strip():
            self.key_frames[self.current_frame_index] = False
            self.essential_frame_checkbox.setChecked(False)
            
        # 更新标注数据中的体重信息
        if self.current_frame_index in self.annotations and self.annotations[self.current_frame_index]:
            # 将体重信息保存到第一个标注中（因为现在每帧只有一个输入框）
            self.annotations[self.current_frame_index][0]['text'] = text
    
    # 更新类别标签高亮状态
    def update_category_labels_highlight(self, selected_annotation):
        """更新类别标签高亮状态"""
        if not hasattr(self, 'category_labels') or not selected_annotation:
            return
    
        for annotation_id, container_widget in self.category_labels.items():
            # 获取标签控件（容器中的第一个子部件）
            if container_widget.layout() and container_widget.layout().count() > 0:
                label_widget = container_widget.layout().itemAt(0).widget()
                if isinstance(label_widget, QLabel):
                    if annotation_id == selected_annotation['id']:
                        # 高亮显示当前聚焦的标注框标签
                        label_widget.setStyleSheet("font-size: 14px; color: white; background-color: #4a90e2; font-family: SimHei, Microsoft YaHei, sans-serif;")
                        container_widget.setStyleSheet("background-color: #4a90e2; padding: 3px 6px; border-radius: 4px;")
                    else:
                        # 恢复默认样式
                        label_widget.setStyleSheet("font-size: 14px; color: #666; background-color: transparent; font-family: SimHei, Microsoft YaHei, sans-serif;")
                        container_widget.setStyleSheet("background-color: transparent;")
    
    # 处理键盘事件
    def keyPressEvent(self, event):
        """处理键盘快捷键"""
        # 获取按键的ASCII码
        key = event.key()
        
        # 'A'/'a'键切换到上一帧
        if key == Qt.Key_A or key == Qt.Key_Left:
            self.prev_frame()
        # 'D'/'d'键切换到下一帧
        elif key == Qt.Key_D or key == Qt.Key_Right:
            self.next_frame()
        # Delete键删除当前高亮标注框
        elif key == Qt.Key_Delete:
            self.delete_annotation()
        # 'R'/'r'重置当前标注框
        elif key == Qt.Key_R:
            self.reset_annotation()
        else:
            # 其他按键调用父类处理
            super().keyPressEvent(event)

    # 设置当前标注工具
    def set_current_tool(self,tool_type):
        """设置当前标注工具"""
        self.current_tool = tool_type
        # 更新按钮样式以显示当前选中的样式
        self.mouse_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'mouse' else "")
        self.rect_tool_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'rectangle' else "")
        # self.point_tool_btn.setStyleSheet("background-color: lightblue;" if tool_type == 'point' else "")
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
                        # 添加标注后更新右侧窗口的类别显示
                        self.update_category_list()
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
                        # 先判断是否在标注框边缘
                        edge_result = self.is_mouse_on_annotation_edge(mapped_point)
                        if edge_result[0]:
                            self.resizing,self.resize_anchor,self.resizing_annotation = edge_result
                        else:
                            inner_result = self.is_mouse_in_annotation_inner(mapped_point)
                            # 如果在标注框内部，选定标注框并高亮
                            if inner_result[0]:
                                self.dragging,self.dragging_annotation,self.drag_offset = inner_result
                                # 选定标注框并高亮
                                self.highlight_annotation(self.dragging_annotation)
                                self.video_display.setCursor(QCursor(Qt.ClosedHandCursor))
                return True
            # 鼠标移动，拖动标注框
            elif event.type() == QEvent.MouseMove:
                mapped_point = self.map_to_original_frame(event.position().toPoint())
                if mapped_point.x() != -1 and mapped_point.y() != -1:
                    self.eventFilter_mouseMoving(mapped_point)
                return True
            # 鼠标释放，结束拖动，更新标注框位置信息
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self.dragging:
                    self.dragging = False
                    # 清除当前拖动的标注框
                    self.dragging_annotation = None
                    # 恢复手掌光标
                    self.video_display.setCursor(QCursor(Qt.OpenHandCursor))
                elif self.resizing:
                    self.resizing = False
                    # 清除当前扩缩的标注框
                    self.resizing_annotation = None
                    # 清除扩缩锚点
                    self.resizing_anchor = None
                    # 恢复默认光标
                    self.video_display.setCursor(QCursor(Qt.ArrowCursor))
                # 重绘当前帧
                self.display_current_frame()
                return True

        return super().eventFilter(obj, event)
    
    # 事件过滤器，处理鼠标移动事件(防止代码冗余，在鼠标移动时调用)
    def eventFilter_mouseMoving(self,mapped_point):
        """处理鼠标移动事件"""
        # 当前处于拖动状态
        if self.dragging:
            # 计算新的标注框位置
            self.mouse_drag(mapped_point)
            # 重绘当前帧
            self.display_current_frame()
            return
        # 当前处于扩缩状态
        if self.resizing:
            # 计算新的标注框位置
            self.mouse_resize(mapped_point)
            # 重绘当前帧
            self.display_current_frame()
            return
        # 处于悬停状态
        if self.current_frame_index in self.annotations:
            # 先判断是否在边缘上
            edge_result,edge_anchor,_ = self.is_mouse_on_annotation_edge(mapped_point)
            if edge_result:
                # 根据边缘位置设置对应的双向箭头光标
                if edge_anchor in ['se','nw']:
                    self.video_display.setCursor(QCursor(Qt.SizeFDiagCursor))
                elif edge_anchor in ['ne','sw']:
                    self.video_display.setCursor(QCursor(Qt.SizeBDiagCursor))
                elif edge_anchor in ['e','w']:
                    self.video_display.setCursor(QCursor(Qt.SizeHorCursor))
                elif edge_anchor in ['n','s']:
                    self.video_display.setCursor(QCursor(Qt.SizeVerCursor))
                return
            inner_result,*_ = self.is_mouse_in_annotation_inner(mapped_point)
            if inner_result:
                self.video_display.setCursor(QCursor(Qt.OpenHandCursor))
                return
        self.video_display.setCursor(QCursor(Qt.ArrowCursor))

    # 重写鼠标按下事件，实现点击页面其他地方时消除输入框焦点
    def mousePressEvent(self, event):
        # 调用父类的mousePressEvent以确保原有功能正常
        super().mousePressEvent(event)

        # 检查当前焦点是否在体重输入框上，如果不在，切换焦点保证快捷键的正常运行
        if self.weight_input.hasFocus():
            # 获取鼠标点击的位置
            click_pos = event.pos()
            # 将输入框的坐标转换为全局坐标进行比较
            input_rect = QRect(self.weight_input.mapToGlobal(self.weight_input.rect().topLeft()), self.weight_input.size())
            
            # 检查点击位置是否在输入框之外
            if not input_rect.contains(click_pos):
                # 清除输入框的焦点
                self.weight_input.clearFocus()

    # 添加closeEvent方法
    def closeEvent(self, event):
        """重写窗口关闭事件，清理资源和终止线程"""
        self.clear_video_resources()
        # 接受关闭事件
        event.accept()
        # 调用父类的关闭事件处理
        super().closeEvent(event)
    
    # 清空视频资源和相关变量
    def clear_video_resources(self):
        """清空视频资源和相关变量"""
        # 清理线程资源
        if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.reader_thread.join(timeout=1.0)
        
        # 清空视频相关变量
        self.video_frames = []
        self.video_path = ""
        self.video_type = False
        self.current_frame_index = 0
        self.total_frame_count = 0
        self.frame_rate = 30  # 重置为默认值
        self.key_frames = {}
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        
        # 清空标注框类别列表
        while self.category_layout.count() > 0:
            item = self.category_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.category_labels.clear()
        
        # 清空标注相关变量
        self.original_annotations = {}
        self.annotations = {}
        
        # 重置绘制和拖动状态
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_color = QColor(Qt.green)  # 重置为默认颜色
        self.current_thickness = 2  # 重置为默认线宽
        self.dragging = False
        self.dragging_annotation = None
        self.drag_offset = QPoint()
        
        # 清空选定标注框信息
        self.selected_annotation = None
        
        # 更新界面状态
        self.loading = False
        self.video_id_value.setText("")
        self.video_display.setText("视频显示区域")  # 重置显示区域文本
        self.statusBar().showMessage("就绪")

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

    # 实现鼠标拖动标注框
    def mouse_drag(self,mapped_point):
        # 计算新的标注框位置
        new_x1 = mapped_point.x() - self.drag_offset.x()
        new_y1 = mapped_point.y() - self.drag_offset.y()
        width = self.dragging_annotation['x2'] - self.dragging_annotation['x1']
        height = self.dragging_annotation['y2'] - self.dragging_annotation['y1']
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height
        # 更新标注框位置
        self.dragging_annotation['x1'] = new_x1
        self.dragging_annotation['y1'] = new_y1
        self.dragging_annotation['x2'] = new_x2
        self.dragging_annotation['y2'] = new_y2
    
    # 实现鼠标调整标注框大小的方法
    def mouse_resize(self, mapped_point):
        # 获取当前标注框的坐标
        x1, y1, x2, y2 = self.resizing_annotation['x1'], self.resizing_annotation['y1'], self.resizing_annotation['x2'], self.resizing_annotation['y2']
        # 根据锚点位置调整标注框大小
        if self.resize_anchor == 'se':     # 右下角
            x2 = mapped_point.x()
            y2 = mapped_point.y()
        elif self.resize_anchor == 'ne':   # 右上角
            x2 = mapped_point.x()
            y1 = mapped_point.y()
        elif self.resize_anchor == 'sw':   # 左下角
            x1 = mapped_point.x()
            y2 = mapped_point.y()
        elif self.resize_anchor == 'nw':   # 左上角
            x1 = mapped_point.x()
            y1 = mapped_point.y()
        elif self.resize_anchor == 'e':  # 右侧
            x2 = mapped_point.x()
        elif self.resize_anchor == 'w':  # 左侧
            x1 = mapped_point.x()
        elif self.resize_anchor == 'n':  # 顶部
            y1 = mapped_point.y()
        elif self.resize_anchor == 's':  # 底部
            y2 = mapped_point.y()
        # 确保new_x1 < new_x2, new_y1 < new_y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # 更新标注框大小
        self.resizing_annotation['x1'] = x1
        self.resizing_annotation['y1'] = y1
        self.resizing_annotation['x2'] = x2
        self.resizing_annotation['y2'] = y2

    # 判断当前鼠标是否在标注框内
    def is_mouse_in_annotation_inner(self,mapped_point):
        for anno in self.annotations[self.current_frame_index]:
            x1, y1, x2, y2 = anno['x1'], anno['y1'], anno['x2'], anno['y2']
            if x1 <= mapped_point.x() <= x2 and y1 <= mapped_point.y() <= y2:
                # 记录当前鼠标位置与左上角的偏移量
                offset = mapped_point - QPoint(x1, y1)
                return True,anno,offset
        return False,None,None

    # 判断当前鼠标是否在标注框的边缘（点击左鼠标时）
    def is_mouse_on_annotation_edge(self,mapped_point):
        """判断当前鼠标是否在标注框的边缘"""
        edge_margin = 10       # 边缘检测的像素宽度
        for anno in self.annotations[self.current_frame_index]:
            x1,y1,x2,y2 = anno['x1'],anno['y1'],anno['x2'],anno['y2']
            mouse_x,mouse_y = mapped_point.x(),mapped_point.y()
            # 检查鼠标是否在右下角边缘(se)
            if (abs(mouse_x - x2) <= edge_margin and abs(mouse_y - y2) <= edge_margin):
                return True,'se',anno
            # 检查鼠标是否在右上角边缘(ne)
            if (abs(mouse_x - x2) <= edge_margin and abs(mouse_y - y1) <= edge_margin):
                return True,'ne',anno
            # 检查鼠标是否在左下角边缘(sw)
            if (abs(mouse_x - x1) <= edge_margin and abs(mouse_y - y2) <= edge_margin):
                return True,'sw',anno
            # 检查鼠标是否在左上角边缘(nw)
            if (abs(mouse_x - x1) <= edge_margin and abs(mouse_y - y1) <= edge_margin):
                return True,'nw',anno
            # 检查是否在右边框上(e)
            if (abs(mouse_x - x2) <= edge_margin and y1 <= mouse_y <= y2):
                return True,'e',anno
            # 检查是否在左边框上(w)
            if (abs(mouse_x - x1) <= edge_margin and y1 <= mouse_y <= y2):
                return True,'w',anno
            # 检查是否在上边框上(n)
            if (abs(mouse_y - y1) <= edge_margin and x1 <= mouse_x <= x2):
                return True,'n',anno
            # 检查是否在下边框上(s)
            if (abs(mouse_y - y2) <= edge_margin and x1 <= mouse_x <= x2):
                return True,'s',anno
        return False,None,None

    # 添加标注框
    def add_annotation(self):
        """添加标注框"""
        # 确保当前帧有视频数据
        if not self.video_frames or self.current_frame_index < 0 or self.current_frame_index >= len(self.video_frames):
            return
        
        # 计算当前帧的最大标注id,从1开始递增
        max_id = 0
        if self.current_frame_index in self.annotations and self.annotations[self.current_frame_index]:
            max_id = max([anno['id'] for anno in self.annotations[self.current_frame_index]])
        annotation_id = max_id + 1

        # 计算矩形坐标（确保x1 < x2, y1 < y2）
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())

        # 设置标注框的标签
        label = f"pig{annotation_id}"

         # 弹出对话框选择猪的类别
        class_items = [f"{class_id}: {class_name}" for class_id, class_name in self.classes.items()]
        if class_items:
            selected_class, ok = QInputDialog.getItem(
                self, "选择猪的类别", "请选择猪的类别:", 
                class_items, 0, False
            )
            if ok and selected_class:
                # 解析选择的类别ID
                class_id = int(selected_class.split(':')[0])
            else:
                # 如果用户取消选择，使用默认类别
                class_id = 0
        else:
            # 如果没有类别数据，使用默认类别
            class_id = 0

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
            'class_id': class_id,
            'text': '',
            # 标注框的颜色，将self.current_color转换为RGB元组
            'color': (self.current_color.red(), self.current_color.green(), self.current_color.blue()),  # 默认是绿色
            # 记录原有颜色
            'original_color': (self.current_color.red(), self.current_color.green(), self.current_color.blue())  # 默认是绿色
        })
    
    # 删除标注框
    def delete_annotation(self,annotation=None):
        # 判断是否有选中的标注框
        if not self.selected_annotation:
            return
        # 如果传入了具体的标注框，则判断是否是选中标注框
        if annotation and annotation != self.selected_annotation:
            return
        # 从当前帧的标注列表中移除选中的标注框
        self.annotations[self.current_frame_index].remove(self.selected_annotation)
        # 清除选中的标注框
        self.selected_annotation = None
        # 刷新当前帧的显示
        self.display_current_frame()
        # 更新类别标签的高亮显示
        self.update_category_list()

    # 高亮显示标注框
    def highlight_annotation(self,annotation):
        """高亮显示标注框"""
        if not annotation:
            return
        # 清除之前的高亮，恢复成原来的颜色
        if self.selected_annotation:
            self.selected_annotation['color'] = self.selected_annotation['original_color']  # 恢复原有颜色
        # 高亮当前标注框
        annotation['color'] = (255, 0, 0)  # 红色
        self.selected_annotation = annotation
        self.display_current_frame()
        self.update_category_labels_highlight(self.selected_annotation)
    
    '''
    # 添加标注框的文本输入框（当前不需要）
    def add_annotation_widget(self, annotation_id, label):
        """添加标注框的文本输入框"""
         # 检查当前帧是否为关键帧，只有关键帧才添加输入框
        if not self.key_frames.get(self.current_frame_index, False):
            return
        
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
        # self.annotations[self.current_frame_index][annotation_id - 1]['text'] = text_input

        # 连接文本变化事件
        # text_input.textChanged.connect(lambda text, aid=annotation_id: self.update_annotation_text(aid, text))
        # 由于我们已经删除了标注框列表，不再需要将输入框添加到布局中
    
    # 更新标注框的文本信息（当前不需要）
    def update_annotation_text(self, annotation_id, text):
        """更新标注框的文本信息"""
        # 检查当前帧是否有标注
        if self.current_frame_index in self.annotation_widgets:
            self.annotations[self.current_frame_index][annotation_id - 1]['text'] = text
    
    # 由于我们已经删除了标注框列表，不再需要清除现有控件或管理布局
    # 关键帧的体重信息现在通过main_content中的weight_input控件管理
    def update_annotation_widgets(self):
        """更新输入框列表组件"""
        # 清除输入框列表原有控件
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
        
        # 只有当前帧时关键帧时，才显示输入框控件
        current_frame_is_essential = self.key_frames.get(self.current_frame_index, False)
        # 如果当前帧不是关键帧，直接返回
        if current_frame_is_essential is False:
            return
        # 如果当前帧还没有在self.annotation_widgets中，先根据当前帧的标注框数量初始化输入框
        if self.current_frame_index not in self.annotation_widgets and self.current_frame_index in self.annotations:
            for i in range(len(self.annotations[self.current_frame_index])):
                self.add_annotation_widget(i+1, f"猪{i+1}")
            return
        
        # 根据每一帧的控件信息添加当前帧的控件
        if current_frame_is_essential and self.current_frame_index in self.annotation_widgets:
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
    '''

    # 保存当前帧的标注信息
    def save_current_frame_annotation(self):
        """保存当前帧的标注信息到txt和json文件"""
        # 检查当前帧是否有关键帧标记
        if not self.key_frames.get(self.current_frame_index, False):
            return
            
        # 检查当前帧是否有标注数据
        if self.current_frame_index not in self.annotations:
            return
            
        try:
            # 获取视频名称作为文件夹名
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            file_name = video_name + '_' + timestamp
            
            # 创建两个不同的目录，分别用于保存txt和json文件
            txt_output_dir = os.path.join(self.config['output_txt_path'], file_name, 'txt')
            json_output_dir = os.path.join(self.config['output_txt_path'], file_name, 'json')
            
            # 创建目录
            if not os.path.exists(txt_output_dir):
                os.makedirs(txt_output_dir)
            if not os.path.exists(json_output_dir):
                os.makedirs(json_output_dir)
                
            # 为当前帧创建单独的TXT文件，文件名格式为frame_xxxxx.txt
            txt_file_name = f"frame_{self.current_frame_index:05d}.txt"
            txt_file_path = os.path.join(txt_output_dir, txt_file_name)
            
            # 为当前帧创建单独的JSON文件，文件名格式为frame_xxxxx.json
            json_file_name = f"frame_{self.current_frame_index:05d}.json"
            json_file_path = os.path.join(json_output_dir, json_file_name)
            
            # 存储单帧JSON数据
            json_single_frame_data = []
            
            # 输出txt文件
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                # 处理当前帧的所有标注信息
                for ann in self.annotations[self.current_frame_index]:
                    x_min, y_min = ann['x1'], ann['y1']
                    x_max, y_max = ann['x2'], ann['y2']
                    class_id = ann['class_id']
                    text = ann['text']
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
                    # 写入JSON数据中
                    json_single_frame_data.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'text': text,
                    })
                    f.write(f"{class_id},{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f},{text}\n")
            
            # 写入单帧JSON文件
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_single_frame_data, f, ensure_ascii=False, indent=4)
                
            # 可以在这里添加日志记录或调试信息
            # print(f"帧 {self.current_frame_index} 的标注已保存")
        except Exception as e:
            # 静默处理错误，不影响用户体验
            print(f"保存当前帧标注时出错: {e}")
    
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
        # 检查是否有视频载入
        if not self.video_path:
            QMessageBox.information(self, "提示", "请先加载视频")
            return
        if self.current_frame_index in self.annotations and self.current_frame_index in self.original_annotations:
            # 将当前帧的标注信息恢复到原始标注字典（作为备份），使用深拷贝创建独立副本
            self.annotations[self.current_frame_index] = copy.deepcopy(self.original_annotations[self.current_frame_index])
            # 清除当前选中的标注框
            self.selected_annotation = None
            # 更新当前帧的图片信息
            self.display_current_frame()
            # 更新当前帧的标注信息
            self.update_category_list()
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
                # 清除之前的高亮，让self.selected_annotation指向一个空对象
                if self.selected_annotation:
                    self.selected_annotation['color'] = self.selected_annotation['original_color']  # 恢复原有颜色
                    self.selected_annotation = None
                self.display_current_frame()
                self.update_frame_info()
            else:
                 QMessageBox.information(self, "提示", "已经是第一帧")
        else:
            QMessageBox.information(self, "提示", "请先加载视频")

    # 切换到下一帧
    def next_frame(self):
        """显示下一帧"""
        # 尝试获取锁，如果获取失败（说明上一次调用还在执行），则直接返回
        if not self.next_frame_lock.acquire(blocking=False):
            return
        try:
            # 检查是否有视频载入
            if not self.video_path:
                QMessageBox.information(self, "提示", "请先加载视频")
                return
            # 检查是否有视频帧
            if self.current_frame_index >= self.total_frame_count - 1:
                # 如果当前还在加载中，等待加载完成
                if self.loading:
                    # 创建一个会自动关闭的消息框
                    QMessageBox.information(self, "提示", "请慢点哦~ 视频君还在加载中")
                    return
                QMessageBox.information(self, "提示", "已经是最后一帧")
                return
            # 如果当前帧的索引号小于self.total_frame_count的长度减一，说明下一帧已经处理，可以使用
            if self.current_frame_index < self.total_frame_count - 1:
                # 如果当前帧的索引号大于等于self.video_frames的长度减1，说明下一帧还没有持久化进self.video_frames，需要从视频帧队列中拉取新的帧信息
                if self.video_type and self.current_frame_index >= len(self.video_frames) - 1:
                    frame = self.frame_queue.get()
                    self.video_frames.append(frame)
                self.current_frame_index += 1
                # 清除之前的高亮
                if self.selected_annotation:
                    self.selected_annotation['color'] = self.selected_annotation['original_color']  # 恢复原有颜色
                    self.selected_annotation = None
                self.display_current_frame()
                self.update_frame_info()
        finally:
            # 无论执行是否成功，都确保释放锁
            self.next_frame_lock.release()

    # 切换到上k帧
    def prev_k_frames(self):
        """显示前k帧"""
        # 检查是否有视频帧
        if hasattr(self, 'video_frames') and self.video_frames:
            # 获取k值
            try:
                k = int(self.k_value_input.text())
                if k <= 0:
                    QMessageBox.information(self, "提示", "k值必须为正整数")
                    return
            except ValueError:
                QMessageBox.information(self, "提示", "请输入有效的k值")
                return
            
            # 计算新的帧索引
            new_index = self.current_frame_index - k
            # 确保不会小于0
            if new_index < 0:
                new_index = 0
                QMessageBox.information(self, "提示", f"已到达第一帧，仅返回 {self.current_frame_index} 帧")
            
            # 更新帧索引并显示
            if new_index != self.current_frame_index:
                self.current_frame_index = new_index
                # 清除之前的高亮
                if self.selected_annotation:
                    self.selected_annotation['color'] = self.selected_annotation['original_color']  # 恢复原有颜色
                    self.selected_annotation = None
                self.display_current_frame()
                self.update_frame_info()
        else:
            QMessageBox.information(self, "提示", "请先加载视频")
            
    # 切换到下k帧
    def next_k_frames(self):
        """显示后k帧"""
        # 尝试获取锁，如果获取失败（说明上一次调用还在执行），则直接返回
        if not self.next_frame_lock.acquire(blocking=False):
            return
        try:
            # 检查是否有视频载入
            if not self.video_path:
                QMessageBox.information(self, "提示", "请先加载视频")
                return
            
            # 获取k值
            try:
                k = int(self.k_value_input.text())
                if k <= 0:
                    QMessageBox.information(self, "提示", "k值必须为正整数")
                    return
            except ValueError:
                QMessageBox.information(self, "提示", "请输入有效的k值")
                return
            
            # 检查是否已到达或超过最后一帧
            if self.current_frame_index >= self.total_frame_count - 1:
                # 如果当前还在加载中，等待加载完成
                if self.loading:
                    # 创建一个会自动关闭的消息框
                    QMessageBox.information(self, "提示", "请慢点哦~ 视频君还在加载中")
                    return
                QMessageBox.information(self, "提示", "已经是最后一帧")
                return
            
            # 计算新的帧索引
            new_index = self.current_frame_index + k
            # 确保不会超过总帧数
            if new_index >= self.total_frame_count - 1:
                new_index = self.total_frame_count - 1
                QMessageBox.information(self, "提示", f"已到达最后一帧，仅前进 {new_index - self.current_frame_index} 帧")
            
            # 更新帧索引并显示
            if new_index != self.current_frame_index:
                # 如果视频帧还没有加载到目标帧，需要确保从队列中加载足够的帧
                if self.video_type and new_index >= len(self.video_frames):
                    # 需要加载的帧数
                    needed_frames = new_index - len(self.video_frames) + 1
                    # 从队列中加载帧
                    for _ in range(needed_frames):
                        if not self.frame_queue.empty():
                            frame = self.frame_queue.get()
                            self.video_frames.append(frame)
                        else:
                            # 如果队列中没有足够的帧，就加载到队列中现有的最后一帧
                            break
                
                # 更新索引并显示帧
                self.current_frame_index = min(new_index, len(self.video_frames) - 1)
                # 清除之前的高亮
                if self.selected_annotation:
                    self.selected_annotation['color'] = self.selected_annotation['original_color']  # 恢复原有颜色
                    self.selected_annotation = None
                self.display_current_frame()
                self.update_frame_info()
        finally:
            # 无论执行是否成功，都确保释放锁
            self.next_frame_lock.release()
    
    # 浏览并选择输入视频路径
    def browse_input_video_path(self):
        """浏览并选择导入的输入图片集路径"""
        directory_path = QFileDialog.getExistingDirectory(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择输入图片集目录", self.config['default_input_path']
        )
        if directory_path:
            self.config['default_input_path'] = directory_path
            self.video_path_input.setText(directory_path)
            self.video_path = directory_path
            self.video_id_value.setText(os.path.basename(directory_path))
            self.load_default_atlas()

    # 浏览并选择输出目录文件夹
    def browse_output_directory(self):
        """浏览并选择输出目录文件夹"""
        directory_path = QFileDialog.getExistingDirectory(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择参数文件输出目录文件夹", self.config['output_txt_path']
        )
        if directory_path:
            self.config['output_txt_path'] = directory_path
            self.output_txt_path_input.setText(directory_path)

    # 浏览并选择输出视频路径
    def browse_video_directory(self):
        """浏览并选择输出视频路径"""
        directory_path = QFileDialog.getExistingDirectory(
            # 当前窗口实例、对话框的标题栏文本、默认文件路径、文件类型过滤器
            self, "选择视频文件输出目录文件夹", self.config['output_video_path']
        )
        if directory_path:
            self.config['output_video_path'] = directory_path
            self.output_video_path_input.setText(directory_path)

# 主运行函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 首先显示功能选择界面
    selection_window = SelectionWindow()
    selection_window.show()
    sys.exit(app.exec())