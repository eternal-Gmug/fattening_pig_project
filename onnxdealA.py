# yolo_onnx_image.py
import cv2
import numpy as np
import onnxruntime as ort
import sys

def load_classes(path):
    """加载类别名称列表"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def letterbox(image, new_shape=1280, color=(114, 114, 114)):
    """resize + padding 保持纵横比"""
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return new_img, r, (dw, dh)

def preprocess(image, input_size):
    img, ratio, dwdh = letterbox(image, new_shape=input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img, ratio, dwdh

def main(onnx_model, image, classes_txt, input_size=1280):
    # 加载类别
    class_names = load_classes(classes_txt)

    # 创建 ONNX Runtime session
    session = ort.InferenceSession(
        onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if image is None:
        print(f"[ERROR] 无法读取图像: {image}")
        return None

    img, ratio, dwdh = preprocess(image, input_size)
    dwdh = np.array([dwdh[0], dwdh[1], dwdh[0], dwdh[1]])

    # 推理结果
    preds = session.run([output_name], {input_name: img})[0]
    preds = np.squeeze(preds)

    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    # 兼容两种常见输出格式
    if preds.shape[1] >= 6:  # 形如 (N,6) 或 (N,7)
        boxes_list = []
        for det in preds:
            if len(det) == 6:
                x0, y0, x1, y1, score, cls_id = det
            else:
                x0, y0, x1, y1, score, conf, cls_id = det
                score = conf  # 用 conf 作为置信度

            if score < 0.3:
                continue

            box = np.array([x0, y0, x1, y1])
            box -= dwdh
            box /= ratio
            box = box.round().astype(np.int32).tolist()

            cls_id = int(cls_id)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            # 四个角点
            x_min, y_min, x_max, y_max = box
            corners = [(x_min, y_min), (x_max, y_min),
                       (x_max, y_max), (x_min, y_max)]
            print(f"类别: {label}, 置信度: {score:.2f}, 角点: {corners}")
            boxes_list.append({
                'cls': cls_id,
                'xyxy': box,    # box参数类型是python列表
                'score': score,
            })
        return boxes_list

    else:
        print(f"[WARN] 模型输出格式不符合预期: {preds.shape}")
        return None


if __name__ == "__main__":

    onnx_model = "model/pig_gesture_best.onnx"
    image_path = "test.jpg"
    classes_txt = "model/classes.txt"

    main(onnx_model, image_path, classes_txt)
