import cv2
import time
import numpy as np
import onnxruntime as ort
import os

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

# ===== CONFIG =====
MODEL_PATH = "/home/pi/Drone_HN/giao_dien_1/best.onnx"
INPUT_SIZE = 320
CONF_THRES = 0.4

# ===== HSV COLORS =====
COLORS = {
    "RED":    ([0, 0, 255],   (0, 0, 255)),
    "BLUE":   ([255, 0, 0],   (255, 0, 0)),
    "YELLOW": ([0, 255, 255], (0, 255, 255))
}

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]

    if hue >= 165:
        lower = np.array([hue - 10, 100, 100], np.uint8)
        upper = np.array([180, 255, 255], np.uint8)
    elif hue <= 15:
        lower = np.array([0, 100, 100], np.uint8)
        upper = np.array([hue + 10, 255, 255], np.uint8)
    else:
        lower = np.array([hue - 10, 100, 100], np.uint8)
        upper = np.array([hue + 10, 255, 255], np.uint8)

    return lower, upper

# ===== ONNX =====
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    outputs = sess.run(None, {input_name: img})[0]
    outputs = outputs.transpose(0, 2, 1)
    pred = outputs[0]

    best = None
    best_conf = 0

    for det in pred:
        if det[4] > best_conf and det[4] > CONF_THRES:
            best = det
            best_conf = det[4]

    if best is not None:
        cx, cy, bw, bh = best[:4]

        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE

        cx = int(cx * scale_x)
        cy = int(cy * scale_y)
        bw = int(bw * scale_x)
        bh = int(bh * scale_y)


        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + bh // 2)


        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("Empty ROI:", x1, y1, x2, y2)
            continue
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        best_color = None
        max_pixels = 0

        for name, (bgr, draw) in COLORS.items():
            lower, upper = get_limits(bgr)
            mask = cv2.inRange(hsv_roi, lower, upper)
            pixels = cv2.countNonZero(mask)

            if pixels > max_pixels:
                max_pixels = pixels
                best_color = (name, draw)

        roi_area = roi.shape[0] * roi.shape[1]
        if best_color and max_pixels / roi_area > 0.25:
            cv2.putText(frame, best_color[0], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, best_color[1], 2)

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO + HSV Pi5", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
