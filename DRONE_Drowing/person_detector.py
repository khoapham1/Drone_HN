# person_detector.py
import cv2
import numpy as np
import onnxruntime as ort

class PersonDetector:
    def __init__(
        self,
        model_path="yolov5n_quant.onnx",
        img_size=320,
        conf_thres=0.5,
        iou_thres=0.45
    ):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load ONNX model
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("✅ Person detection model loaded (YOLOv5 + NMS enabled)")

    # ---------------- LETTERBOX ----------------
    def letterbox(self, img, new_shape=(320, 320), color=(114, 114, 114)):
        shape = img.shape[:2]  # h, w
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = (
            int(round(shape[1] * r)),
            int(round(shape[0] * r))
        )

        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img,
            int(round(dh - 0.1)), int(round(dh + 0.1)),
            int(round(dw - 0.1)), int(round(dw + 0.1)),
            cv2.BORDER_CONSTANT,
            value=color
        )
        return img, r, dw, dh

    # ---------------- DETECT ----------------
    def detect(self, frame):
        img, r, dw, dh = self.letterbox(
            frame, (self.img_size, self.img_size)
        )

        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0

        # Inference
        pred = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0]

        boxes = []
        scores = []

        # YOLOv5 output processing
        for x in pred:
            x = x[x[:, 4] > self.conf_thres]
            if len(x) == 0:
                continue

            # obj_conf * class_conf
            x[:, 5:] *= x[:, 4:5]

            # xywh -> xyxy
            box = np.zeros((x.shape[0], 4))
            box[:, 0] = x[:, 0] - x[:, 2] / 2
            box[:, 1] = x[:, 1] - x[:, 3] / 2
            box[:, 2] = x[:, 0] + x[:, 2] / 2
            box[:, 3] = x[:, 1] + x[:, 3] / 2

            conf = x[:, 5:].max(1)
            cls = x[:, 5:].argmax(1)

            # Chỉ giữ PERSON (class 0)
            keep = (cls == 0)
            box = box[keep]
            conf = conf[keep]

            for i in range(len(box)):
                x1, y1, x2, y2 = box[i]

                # scale back to original image
                x1 = int((x1 - dw) / r)
                y1 = int((y1 - dh) / r)
                x2 = int((x2 - dw) / r)
                y2 = int((y2 - dh) / r)

                # clamp
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(conf[i]))

        # ---------------- NMS ----------------
        detections = []

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                self.conf_thres,
                self.iou_thres
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': scores[i],
                        'center_x': x + w // 2,
                        'center_y': y + h // 2,
                        'width': w,
                        'height': h
                    })

        return detections
