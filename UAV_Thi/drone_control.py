import cv2
import time
import numpy as np
from dronekit import connect, VehicleMode
from pymavlink import mavutil

from yolodetector import YOLODetector
from hsv_color import HSVColorClassifier
from CenterTracker import CenterTracker
from PID_control import PIDController
import subprocess


# ================= CONFIG =================
TARGET_ALT = 1.5
MODEL_PATH = "/home/pi/Drone_HN/giao_dien_1/best.onnx"

HOLD_TIME = 5          # Giữ 2 giây
ERROR_THRESH = 60      # deadzone (pixel)
VISION_TIMEOUT = 10.0     # mất vision quá lâu → LAND
DEBUG_MODE = True         # Bật chế độ debug
servo_triggered = {
    1: False,
    3: False,
    5: False
}


# ==== COMPASS =====
TARGET_COMPASS_HEADING = 13
COMPASS_TOLERANCE  = 5
YAW_SPEED = 30

# ================= PID =================
# Giảm hệ số PID để tránh overshoot
PID_X = PIDController(0.00008, 0.0, 0.0, max_output=0.3)  # Giảm Kp
PID_Y = PIDController(0.00008, 0.0, 0.0, max_output=0.3)

# ================= GLOBAL MISSION VARIABLES =================
mission_step = 0
hold_start_time = 0
in_target_zone = False
vehicle = None

# ================= CONTROL FUNCTIONS =================
def arm_and_takeoff(target_altitude):
    print("Basic pre-arm checks")
    while vehicle.mode.name != 'GUIDED':
        print('Waiting for GUIDED...')
        time.sleep(1)
    print("Arming motors")

    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)
    time.sleep(1)
    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        current_altitude = vehicle.rangefinder.distance
        print(" Altitude: ", current_altitude)
        if current_altitude >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def land():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    while vehicle.armed:
        print(" Waiting for landing...")
        time.sleep(1)
    print("Landed and disarmed.")
    
def send_local_ned_velocity(vx, vy, vz=0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        vehicle._master.target_system,
        vehicle._master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        0b0000111111000111,                  
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def condition_yaw(heading, relative=False):
    """
    Gửi lệnh xoay drone đến heading cụ thể
    heading: Góc heading mong muốn (độ)
    relative: Nếu True, heading là tương đối so với hướng hiện tại
    """
    if relative:
        is_relative = 1  # góc tương đối
    else:
        is_relative = 0  # góc tuyệt đối
    
    # Tạo lệnh MAV_CMD_CONDITION_YAW
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target_system, target_component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,       # confirmation
        heading, # param 1: góc yaw (độ)
        YAW_SPEED, # param 2: tốc độ yaw (độ/giây)
        1,       # param 3: hướng (-1: counter clockwise, 1: clockwise)
        is_relative, # param 4: góc tương đối (1) hay tuyệt đối (0)
        0, 0, 0  # param 5-7: không sử dụng
    )
    
    # Gửi lệnh
    vehicle.send_mavlink(msg)
    vehicle.flush()
    print(f"[COMPASS] Sent yaw command to {heading}° {'relative' if relative else 'absolute'} at {YAW_SPEED}°/s")

def xoay_compass(target_heading=TARGET_COMPASS_HEADING):
    """
    Xoay drone đến hướng compass cụ thể
    target_heading: Hướng compass mong muốn (0-360 độ)
    """
    print(f"[COMPASS] Starting compass rotation to {target_heading}°")
    
    # Gửi lệnh xoay đến heading tuyệt đối
    condition_yaw(target_heading, relative=False)
    
    # Chờ cho đến khi đạt được heading mong muốn
    start_time = time.time()
    timeout = 15  # Giới hạn thời gian tối đa 15 giây
    
    while True:
        # Lấy heading hiện tại của drone
        current_heading = vehicle.heading
        
        # Tính độ lệch giữa heading hiện tại và target
        heading_diff = abs(current_heading - target_heading)
        
        # Xử lý trường hợp vòng tròn (ví dụ: 350° và 10°)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff
        
        print(f"[COMPASS] Current: {current_heading}°, Target: {target_heading}°, Diff: {heading_diff:.1f}°")
        
        # Kiểm tra xem đã đạt được heading chưa
        if heading_diff <= COMPASS_TOLERANCE:
            print(f"[COMPASS] Reached target heading {target_heading}° (current: {current_heading}°)")
            break
        
        # Kiểm tra timeout
        if time.time() - start_time > timeout:
            print(f"[COMPASS WARNING] Timeout after {timeout}s. Current: {current_heading}°, Target: {target_heading}°")
            break
        
        time.sleep(0.1)
    
    # Dừng xoay
    condition_yaw(vehicle.heading, relative=False)
    print("[COMPASS] Compass rotation completed")

def move_with_timer(direction, duration, speed=0.3):

    vx, vy = 0, 0
    
    safe_speed = min(speed, 0.5)
    
    if direction.lower() == 'forward':
        vx = safe_speed 
    elif direction.lower() == 'backward':
        vx = -safe_speed  
    elif direction.lower() == 'left':
        vy = -safe_speed  
    elif direction.lower() == 'right':
        vy = safe_speed 
    else:
        print(f"Unknown direction: {direction}")
        return
    
    print(f"[MOVE] {direction} for {duration}s at {safe_speed:.1f}m/s")
    
    start_time = time.time()
    end_time = start_time + duration
    
    try:
        if direction.lower() in ['left', 'right']:
            current_alt = vehicle.location.global_relative_frame.alt
            if current_alt < 0.5:  # Du?i 0.5m th� kh�ng di chuy?n ngang
                print(f"[WARNING] Altitude too low ({current_alt:.1f}m) for strafing")
                return
    except:
        pass
    
    while time.time() < end_time:
        send_local_ned_velocity(vx, vy, 0)
        time.sleep(0.05)  # 20Hz
    
    send_local_ned_velocity(0, 0, 0)
    print(f"[MOVE] Completed {direction} movement")
    
    time.sleep(0.2)

def control_drone_to_center(error_x, error_y):
    """Điều khiển drone về tâm circle"""
    # Tính toán velocity với deadzone
    if abs(error_x) < 5 and abs(error_y) < 5:
        # Nếu error rất nhỏ, không điều khiển
        send_local_ned_velocity(0, 0, 0)
        return
    
    vx = PID_Y.update(-error_y)  # error_y điều khiển vx
    vy = PID_X.update(error_x)   # error_x điều khiển vy
    
    # Clamp velocity để tránh quá lớn
    vx = max(min(vx, 0.3), -0.3)
    vy = max(min(vy, 0.3), -0.3)
    
    print(f"[PID] Error: ({error_x:.1f}, {error_y:.1f}) -> Velocity: ({vx:.3f}, {vy:.3f})")
    send_local_ned_velocity(vx, vy, 0)

def is_in_target_zone(error_x, error_y):
    """Kiểm tra xem drone đã ở trong vùng target chưa"""
    return abs(error_x) < ERROR_THRESH and abs(error_y) < ERROR_THRESH

def reset_mission_state():
    """Reset trạng thái mission"""
    global hold_start_time, in_target_zone
    hold_start_time = 0
    in_target_zone = False
    PID_X.reset()
    PID_Y.reset()

def run_servo(sig):
    subprocess.Popen(
        ["/usr/bin/python3", "co_cau.py", str(sig)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ================= MISSION STATE MACHINE =================
def mission_state_machine(error_x, error_y, detected_color, frame=None):
    global servo_triggered
    """State machine điều khiển mission"""
    global mission_step, hold_start_time, in_target_zone
    
    print(f"[STATE {mission_step}] Color: {detected_color}, Error: ({error_x:.1f}, {error_y:.1f})")
    
    # Mission Step 0: Takeoff và di chuyển forward
    if mission_step == 0:

        mission_step = 1

        # move_with_timer("Right", 6, 0.5)
        # print("[MISSION] Reached Step 3 Move Right")
        land()
        print("[MISSION] Reached Step 4 Land")
# ================= MISSION CLASS =================
class Mission:
    def __init__(self):
        self.detector = YOLODetector(MODEL_PATH, conf_thres=0.8)
        self.color = HSVColorClassifier(color_threshold=0.05, debug=True)
        self.tracker = CenterTracker(offset=(0, 0), circle_detect=250)
        
        self.last_detect_time = time.time()
        self.no_detection_count = 0
        self.MAX_NO_DETECTION = 30

    def get_better_roi(self, frame, bbox):
        """Lấy ROI tốt hơn - lấy toàn bộ bbox"""
        x1, y1, x2, y2 = bbox["bbox"]
        
        # Mở rộng bbox một chút để đảm bảo lấy đủ hình tròn
        expand = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w, x2 + expand)
        y2 = min(h, y2 + expand)
        
        return frame[y1:y2, x1:x2]

    def draw_detections(self, frame, bbox, color_name, error):
        """Vẽ thông tin detection lên frame"""
        if bbox is None or color_name is None:
            return
        
        x1, y1, x2, y2 = bbox["bbox"]
        cx, cy = bbox["center"]
        conf = bbox.get("conf", 0.0)
        
        # Vẽ bbox
        color_map = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "BLUE": (255, 0, 0)
        }
        draw_color = color_map.get(color_name, (0, 255, 0))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 1)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), 60)
        
        # Vẽ label
        label = f"{color_name} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
        
        # Vẽ error
        if error:
            ex, ey = error
            cv2.putText(frame, f"ex: {ex:.1f}, ey: {ey:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Vẽ tracking circle và center
        self.tracker.draw_center(frame)
        
        # Vẽ mission state
        cv2.putText(frame, f"Mission Step: {mission_step}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        global mission_step
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prev_time = time.time()
        
        print(f"[MISSION] Starting mission from step {mission_step}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame lost")
                time.sleep(0.1)
                continue
            
            # # Kiểm tra timeout vision
            # if time.time() - self.last_detect_time > VISION_TIMEOUT and mission_step > 0:
            #     print("[WARN] Vision timeout -> LAND")
            #     land()
            #     break
            
            # Phát hiện vật thể
            bbox = self.detector.detect_best(frame)
            
            if bbox is None:
                self.no_detection_count += 1
                if self.no_detection_count > self.MAX_NO_DETECTION:
                    print("[WARN] No detection for too long")
                    send_local_ned_velocity(0, 0, 0)
                    PID_X.reset()
                    PID_Y.reset()
                    reset_mission_state()
                
                # Hiển thị frame không có detection
                now = time.time()
                fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "NO DETECTION", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                self.tracker.draw_center(frame)
                cv2.imshow("MISSION", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Reset counter khi phát hiện được
            self.no_detection_count = 0
            self.last_detect_time = time.time()
            
            try:
                x1, y1, x2, y2 = bbox["bbox"]
                cx, cy = bbox["center"]
                conf = bbox.get("conf", 0.0)
                
                # Kiểm tra tính hợp lệ
                h, w = frame.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
                    print(f"[WARN] Invalid bbox")
                    continue
                    
            except (KeyError, TypeError) as e:
                print(f"[ERROR] Invalid bbox format: {e}")
                continue
            
            # Lấy ROI (toàn bộ bbox)
            roi = self.get_better_roi(frame, bbox)
            if roi.size == 0:
                print("[WARN] Empty ROI")
                continue
            
            # Phân loại màu
            color_result = self.color.classify(roi)
            
            if color_result is None:
                if DEBUG_MODE:
                    print(f"[DEBUG] Cannot classify color")
                continue
            
            color_name, _ = color_result
            
            # Tính toán error
            error = self.tracker.compute_error(frame, (x1, y1, x2, y2))
            if error is None:
                print("[WARN] Object out of tracking circle")
                continue
            
            ex, ey = error
            
            # Chạy state machine
            state_result = mission_state_machine(ex, ey, color_name, frame)
            
            # Vẽ thông tin detection
            self.draw_detections(frame, bbox, color_name, (ex, ey))
            
            # Hiển thị FPS
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"State: {state_result}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Kiểm tra mission hoàn thành
            if mission_step >= 6:
                cv2.putText(frame, "MISSION COMPLETED", (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow("MISSION", frame)
                cv2.waitKey(3000)
                break
            
            cv2.imshow("MISSION", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# ================= MAIN =================
if __name__ == "__main__":
    try:
        vehicle = connect('/dev/ttyACM0', wait_ready=True, baud=115200)
        print("Connect successfully /dev/ttyACM0")
    except Exception as e:
        print(f"[ERROR] Cannot connect to vehicle: {e}")
        print("[INFO] Running in simulation mode (no vehicle connection)")
        vehicle = None
    
    try:
        print("[MISSION] Starting Mission")
        print("[MISSION] Step 0: Taking off and moving forward 6.5m")
        arm_and_takeoff(TARGET_ALT)
        time.sleep(1)
        # xoay_compass(TARGET_COMPASS_HEADING)
        # time.sleep(1)
        move_with_timer("backward", 5, 0.5)
        print("[MISSION] Reached Step 1 Move Forward")
        move_with_timer("forward", 5, 0.5)
        print("[MISSION] Reached Step 2 Move Left")

        move_with_timer("right", 5, 0.5)
        print("[MISSION] Reached Step 3 Move right")
        land()
        print("LANDED....")

        # mission = Mission()
        # mission.run()

        
    except KeyboardInterrupt:
        print("\n[INFO] Mission interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vehicle is not None:
            vehicle.close()
            print("[INFO] Vehicle closed")
        else:
            print("[INFO] Simulation ended")
