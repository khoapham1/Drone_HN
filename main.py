import cv2
import time
import math
import numpy as np
from collections import deque
import mediapipe as mp
from person_detector import PersonDetector

# ================= CONFIG =================
VIDEO_PATH = "video_3.mp4"   # đổi sang 0 nếu dùng webcam
# VIDEO_PATH = 0  # Sử dụng webcam
IMG_SIZE = 320

SOS_FRAMES = 40
WARNING_FRAMES = 20

ARM_SPEED_TH = 8
BODY_MOVE_TH = 4
HEAD_SHOULDER_RATIO_TH = 0.9  # Tỷ lệ đầu-vai
FACE_WATER_TH = 0.15  # Mặt ở dưới nước
VERTICAL_ANGLE_TH = 30  # Góc nghiêng cơ thể

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# ================= UTILS =================
def lm_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def calculate_angle(a, b, c):
    """Tính góc giữa 3 điểm"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def calculate_body_tilt(shoulder_left, shoulder_right, hip_left, hip_right):
    """Tính độ nghiêng cơ thể"""
    shoulder_center = ((shoulder_left[0] + shoulder_right[0]) // 2,
                      (shoulder_left[1] + shoulder_right[1]) // 2)
    hip_center = ((hip_left[0] + hip_right[0]) // 2,
                 (hip_left[1] + hip_right[1]) // 2)
    
    # Tính góc so với phương thẳng đứng
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]
    
    if dy == 0:
        return 90
    
    angle = math.degrees(math.atan(abs(dx) / abs(dy)))
    return angle

# ================= TRACK EXT =================
def init_track(track):
    if hasattr(track, "inited"):
        return
    track.inited = True
    track.left_wrist = deque(maxlen=30)
    track.right_wrist = deque(maxlen=30)
    track.center_hist = deque(maxlen=30)
    track.head_shoulder_ratio_hist = deque(maxlen=30)
    track.body_tilt_hist = deque(maxlen=30)
    track.face_water_hist = deque(maxlen=15)
    track.sos_cnt = 0
    track.warning_cnt = 0
    track.state = "ACTIVE"
    track.sos_start_time = None

# ================= MAIN =================
def main():
    # Sửa lỗi Qt platform plugin trên Raspberry Pi
    import os
    os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Sử dụng XCB thay vì Wayland
    
    detector = PersonDetector("yolov5n_quant.onnx", IMG_SIZE, 0.25, 0.45)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Lấy FPS của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 1.0/30.0

    prev = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)

        detections = detector.detect(frame)

        for det in detections:
            t = det["track_obj"]
            init_track(t)

            x1, y1, x2, y2 = t.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            t.center_hist.append((cx, cy))

            if not pose_res.pose_landmarks:
                continue

            lm = pose_res.pose_landmarks.landmark

            # Lấy các điểm quan trọng
            nose = lm_xy(lm[0], w, h)
            left_eye = lm_xy(lm[1], w, h)
            right_eye = lm_xy(lm[2], w, h)
            left_shoulder = lm_xy(lm[11], w, h)
            right_shoulder = lm_xy(lm[12], w, h)
            left_hip = lm_xy(lm[23], w, h)
            right_hip = lm_xy(lm[24], w, h)
            
            lw = lm_xy(lm[15], w, h)
            rw = lm_xy(lm[16], w, h)

            t.left_wrist.append(lw)
            t.right_wrist.append(rw)

            # ================= CÁC ĐIỀU KIỆN PHÁT HIỆN =================
            
            # 1. TAY TRÊN ĐẦU - CẢI TIẾN
            head_center_y = (nose[1] + left_eye[1] + right_eye[1]) / 3
            arm_above_head = (lw[1] < head_center_y) or (rw[1] < head_center_y)
            
            # 2. TỐC ĐỘ TAY - CẢI TIẾN
            def arm_speed(hist):
                if len(hist) < 2:
                    return 0
                speeds = []
                for i in range(1, len(hist)):
                    d = dist(hist[i], hist[i-1])
                    speeds.append(d)
                return np.mean(speeds) if speeds else 0
            
            arm_v = (arm_speed(t.left_wrist) + arm_speed(t.right_wrist)) / 2
            arm_active = arm_v > ARM_SPEED_TH
            
            # 3. CƠ THỂ BẤT ĐỘNG - CẢI TIẾN
            # SỬA LỖI: Không dùng slice trực tiếp trên deque
            if len(t.center_hist) > 5:
                # Chuyển deque sang list trước khi slice
                center_list = list(t.center_hist)
                # Lấy 10 frame gần nhất, nhưng đảm bảo không vượt quá độ dài hiện có
                start_idx = max(0, len(center_list) - 10)
                recent_centers = center_list[start_idx:]
                
                xs = [p[0] for p in recent_centers]
                ys = [p[1] for p in recent_centers]
                body_move = np.std(xs) + np.std(ys) if len(xs) > 1 else 999
            else:
                body_move = 999

            frozen = body_move < BODY_MOVE_TH
            
            # 4. TỶ LỆ ĐẦU-VAI (dấu hiệu đầu chìm)
            shoulder_width = dist(left_shoulder, right_shoulder)
            head_height = abs(nose[1] - (left_shoulder[1] + right_shoulder[1]) / 2)
            
            if shoulder_width > 0:
                head_shoulder_ratio = head_height / shoulder_width
                t.head_shoulder_ratio_hist.append(head_shoulder_ratio)
                
                # Nếu tỷ lệ đầu-vai nhỏ trong 15 frame liên tiếp
                if len(t.head_shoulder_ratio_hist) >= 15:
                    # Sửa lỗi: Chuyển deque sang list
                    ratio_list = list(t.head_shoulder_ratio_hist)
                    low_ratio_frames = sum(1 for r in ratio_list 
                                         if r < HEAD_SHOULDER_RATIO_TH)
                    head_submerged = low_ratio_frames >= 10
                else:
                    head_submerged = False
            else:
                head_submerged = False
            
            # 5. GÓC NGHIÊNG CƠ THỂ (dấu hiệu mất thăng bằng)
            body_tilt = calculate_body_tilt(left_shoulder, right_shoulder, left_hip, right_hip)
            t.body_tilt_hist.append(body_tilt)
            
            if len(t.body_tilt_hist) >= 10:
                # Sửa lỗi: Chuyển deque sang list trước khi slice
                tilt_list = list(t.body_tilt_hist)
                start_idx = max(0, len(tilt_list) - 10)
                recent_tilts = tilt_list[start_idx:]
                avg_tilt = np.mean(recent_tilts)
                body_unstable = avg_tilt > VERTICAL_ANGLE_TH
            else:
                body_unstable = False
            
            # 6. MẶT Ở DƯỚI NƯỚC (nếu có thông tin về mặt nước)
            # Giả định mực nước ở 2/3 chiều cao người từ chân lên
            water_level = y1 + (y2 - y1) * 2/3
            face_in_water = nose[1] > water_level
            
            t.face_water_hist.append(face_in_water)
            if len(t.face_water_hist) >= 10:
                # Sửa lỗi: Chuyển deque sang list
                water_list = list(t.face_water_hist)
                face_water_frames = sum(1 for f in water_list if f)
                prolonged_face_water = face_water_frames >= 8
            else:
                prolonged_face_water = False

            # ================= QUYẾT ĐỊNH ĐA TIÊU CHÍ =================
            primary_condition = arm_above_head and arm_active and frozen
            secondary_condition = head_submerged and body_unstable
            tertiary_condition = prolonged_face_water and frozen
            
            if primary_condition:
                t.sos_cnt += 2  # Tăng nhanh hơn cho điều kiện chính
            elif secondary_condition or tertiary_condition:
                t.sos_cnt += 1
            else:
                t.sos_cnt = max(0, t.sos_cnt - 1)
            
            # Phân loại trạng thái
            if t.sos_cnt > SOS_FRAMES:
                t.state = "SOS"
                if t.sos_start_time is None:
                    t.sos_start_time = time.time()
            elif t.sos_cnt > WARNING_FRAMES:
                t.state = "WARNING"
                t.sos_start_time = None
            else:
                t.state = "ACTIVE"
                t.sos_start_time = None

            # ================= VẼ KẾT QUẢ =================
            color = (0, 255, 0)
            if t.state == "WARNING":
                color = (0, 165, 255)
            if t.state == "SOS":
                color = (0, 0, 255)
                # Hiển thị thời gian SOS
                if t.sos_start_time:
                    duration = time.time() - t.sos_start_time
                    cv2.putText(frame, f"SOS: {duration:.1f}s", 
                               (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ đường mực nước
            water_y = int(water_level)
            cv2.line(frame, (x1, water_y), (x2, water_y), 
                    (255, 255, 0), 1, cv2.LINE_AA)
            
            # Thông tin chi tiết (chỉ hiển thị nếu có đủ không gian)
            info_y = y2 + 20
            info_lines = []
            
            if (y2 + 100) < h:  # Đảm bảo không vẽ ra ngoài frame
                info_lines = [
                    f"ArmAbove: {arm_above_head}",
                    f"ArmSpeed: {arm_v:.1f}",
                    f"Frozen: {frozen}",
                    f"HeadRatio: {head_shoulder_ratio:.2f}" if shoulder_width > 0 else "HeadRatio: N/A",
                    f"Tilt: {body_tilt:.1f}°",
                    f"FaceWater: {face_in_water}"
                ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (x1, info_y + i*15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            # Vẽ skeleton với màu tương ứng trạng thái
            mp_draw.draw_landmarks(
                frame, pose_res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
            )
            
            # Vẽ điểm đầu
            cv2.circle(frame, nose, 3, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {t.id} | {t.state}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS và thống kê
        current_time = time.time()
        fps = 1 / (current_time - prev) if (current_time - prev) > 0 else 0
        prev = current_time
        
        # Thống kê số người trong các trạng thái
        states = {"ACTIVE": 0, "WARNING": 0, "SOS": 0}
        for det in detections:
            t = det["track_obj"]
            if hasattr(t, 'state'):
                states[t.state] += 1
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Active: {states['ACTIVE']} | Warn: {states['WARNING']} | SOS: {states['SOS']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Thử hiển thị trên cửa sổ
        try:
            cv2.imshow("DROWNING DETECTION - IMPROVED", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"Không thể hiển thị cửa sổ: {e}")
            # Lưu frame ra file nếu không hiển thị được
            if frame_count % 30 == 0:  # Lưu mỗi 30 frame
                cv2.imwrite(f"output_frame_{frame_count}.jpg", frame)
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()