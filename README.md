# 育猪行为标注工具

## 项目简介

育猪行为标注工具是一款基于Python和PySide6开发的专业标注软件，用于对猪只的行为和姿态进行高效标注和分析。该工具集成了ONNX模型推理功能，能够自动检测视频或图像中的猪只，并支持三种不同的猪只姿态识别（站立背部、侧面和躺卧姿态）。

## 主要功能

### 方框标注工具 (BoxAnnotationTool)

- **视频/图片加载**：支持加载视频文件和图像文件进行标注
- **AI自动标注**：集成ONNX模型推理，自动检测图像中的猪只位置和姿态
- **手动标注编辑**：提供直观的界面进行手动绘制、调整和删除标注框
- **帧导航**：支持视频帧的前进、后退和固定帧跳转
- **标注框管理**：支持选择、移动、调整大小和删除标注框
- **结果导出**：支持将标注结果导出为JSON和TXT格式，便于后续分析和处理

## 技术栈

- **编程语言**：Python
- **GUI框架**：PySide6
- **图像处理**：OpenCV (opencv-python)
- **数值计算**：NumPy
- **模型推理**：ONNX Runtime
- **深度学习模型**：YOLOv8m

## 项目结构

```
d:\Project\Python\fattening_pig_project/
├── annotation_tool.py     # 标注工具主程序，包含BoxAnnotationTool和SegmentationAnnotationTool类
├── onnxdeal.py            # 模型推理模块
├── onnxdealA.py           # 增强版模型推理模块，支持CPU和CUDA
├── model/                 # 模型存储目录
│   ├── classes.txt        # 类别定义文件
│   ├── yolov8m_gray.onnx  # 灰度模型文件
│   ├── pig_gesture_best.onnx  # 姿态识别模型
├── input_atlas/           # 输入图像文件夹示例
├── output/                # 标注结果输出目录
├── requirements.txt       # 项目依赖文件
├── annotation_tool.spec   # PyInstaller打包配置文件
└── dist/                  # 打包后的可执行文件目录
```

## 安装说明

### 环境要求

- Python 3.7+
- Windows/Linux/macOS 操作系统

### 依赖安装

1. 克隆或下载项目到本地
2. 安装项目依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行程序

```bash
python annotation_tool.py
```

或者使用已打包的可执行文件：

```bash
./dist/annotation_tool.exe
```

### 主要操作步骤

1. **加载文件**：
   - 点击菜单栏的文件选项
   - 选择打开视频或图像
   - 或者使用工具栏中的加载按钮

2. **自动标注**：
   - 加载完文件后，系统会自动调用ONNX模型对当前帧进行自动检测和标注

3. **手动编辑标注**：
   - 在鼠标拖动模式下，点击已有标注框进行选择
   - 拖动边框调整大小
   - 拖动标注框移动位置
   - 使用类别标签旁的删除按钮删除标注

4. **帧导航**：
   - 使用工具栏中的前进、后退按钮导航视频帧
   - 能够按照固定间隔帧进行跳转

5. **保存结果**：
   - 点击菜单栏的保存选项
   - 结果将保存到output目录中，包含JSON和TXT格式

## 模型说明

### 支持的猪只姿态类别

- `pig_back`：猪只站立时的背部视图
- `pig_side`：猪只站立时的侧面视图
- `pig_lying`：猪只躺卧姿态

### 模型文件

- `yolov8m_gray.onnx`：针对灰度图像优化的YOLOv8m模型
- `pig_gesture_best.onnx`：猪只姿态识别的最佳模型

## 开发说明

### 扩展功能

- 如需添加新的猪只姿态类别，请修改 `model/classes.txt` 文件
- 如需更新模型，请替换 `model/` 目录下的ONNX文件

### 打包应用

项目已配置PyInstaller打包脚本，可以生成独立的可执行文件：

```bash
pyinstaller annotation_tool.spec
```

## 注意事项与错误处理

1. **确保模型文件存在**：首次运行前，请确保 `model/` 目录下包含所需的ONNX模型文件
2. **CUDA加速**：如果系统支持CUDA，模型推理将自动使用GPU加速以提高性能
3. **标注框删除**：删除标注框时，需先选中要删除的标注框
4. **帧索引检查**：程序会进行帧索引有效性检查，避免访问无效帧
5. **异常处理**：程序包含异常处理机制，出现错误时会显示提示信息
