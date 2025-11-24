#!/usr/bin/env python3
import time
import math
import threading
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from picamera2 import Picamera2
import requests
from PID_controller import PIDController

# ---- camera / aruco params ----
aruco = cv2.aruco
ids_to_find = [1, 2]

# find_aruco is now dynamic, will be set via class
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # legacy API
parameters = cv2.aruco.DetectorParameters_create()              # legacy API
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# Adaptive threshold cho vùng sáng/tối không đều
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 33
parameters.adaptiveThreshWinSizeStep = 10

# Giảm yêu cầu kích thước marker
parameters.minMarkerPerimeterRate = 0.02
parameters.maxMarkerPerimeterRate = 4.0

# Giới hạn tỷ lệ cạnh để tránh blob méo
parameters.minCornerDistanceRate = 0.05
parameters.minMarkerDistanceRate = 0.05

# Tinh chỉnh threshold
parameters.adaptiveThreshConstant = 7

# ===== PID CONTROLLERS (nếu cần dùng ArUco tracking) =====
PID_X = PIDController(Kp=0.002, Ki=0.0007, Kd=0.0, max_output=0.5)  # m/s
PID_Y = PIDController(Kp=0.002, Ki=0.0007, Kd=0.0, max_output=0.5)  # m/s

horizontal_res = 1280
vertical_res = 720
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

calib_path = "/home/pi/Project_drone_delivery-main/"
np_camera_matrix = np.load(calib_path + 'camera_matrix_gpt.npy')
np_dist_coeff = np.load(calib_path + 'dist_coeff_gpt.npy')

time_to_wait = 0.1
time_last = 0

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_picamera2 = None
_camera_thread = None
_camera_running = False


# ==================== CAMERA FUNCTIONS ====================

def start_camera():
    """
    Start Picamera2 and continuously capture frames
    """
    global _picamera2, _camera_running, _camera_thread

    if _camera_running:
        return

    try:
        _picamera2 = Picamera2()

        config = _picamera2.create_video_configuration(
            main={
                "size": (1280, 720),
                "format": "RGB888"
            },
            controls={
                # Có thể bật/tinh chỉnh thêm nếu cần
                # "AwbEnable": True,
                # "AwbMode": 0,
                # "Brightness": 0,
                # "Contrast": 1,
                # "Saturation": 0.5,
                # "Sharpness": 0,
            }
        )
        _picamera2.configure(config)
        _picamera2.start()

        _camera_running = True
        _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
        _camera_thread.start()
        print("Started Picamera2 successfully with optimized color settings")

    except Exception as e:
        print("Failed to start Picamera2:", e)


def _camera_loop():
    """
    Continuously capture frames and update latest frame
    """
    global _latest_frame_jpeg, _latest_frame_lock, _camera_running

    while _camera_running and _picamera2:
        try:
            frame = _picamera2.capture_array()

            if frame is not None:
                # Encode JPEG (chất lượng cao)
                ret, jpeg = cv2.imencode(
                    '.jpg', frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90,
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
                )
                if ret:
                    with _latest_frame_lock:
                        _latest_frame_jpeg = jpeg.tobytes()

            time.sleep(0.033)  # ~30 fps
        except Exception as e:
            print("Error in camera loop:", e)
            time.sleep(0.1)


def stop_camera():
    """
    Stop camera and clean up
    """
    global _picamera2, _camera_running, _camera_thread

    _camera_running = False
    if _camera_thread:
        _camera_thread.join(timeout=2.0)

    if _picamera2:
        try:
            _picamera2.stop()
            _picamera2 = None
        except Exception as e:
            print("Error stopping camera:", e)

    print("Stopped camera")


def get_lastest_frame():
    """
    Return latest JPEG bytes or None.
    """
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg


def preprocess_aruco_image(bgr):
    """
    Tiền xử lý ảnh cho ArUco:
    - CLAHE + auto gamma
    - Bilateral filter
    Trả về ảnh GRAY uint8.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Auto gamma theo độ sáng
    mean_intensity = float(np.mean(gray))
    gamma = 1.0
    if mean_intensity < 60:
        gamma = 0.5
    elif mean_intensity < 100:
        gamma = 0.7
    elif mean_intensity > 200:
        gamma = 1.8
    elif mean_intensity > 170:
        gamma = 1.4

    if abs(gamma - 1.0) > 1e-3:
        gray_norm = gray.astype(np.float32) / 255.0
        gray_gamma = np.power(gray_norm, gamma)
        gray = np.clip(gray_gamma * 255.0, 0, 255).astype(np.uint8)

    # Lọc nhiễu nhưng giữ biên
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    # Chuẩn hóa lại full dynamic range
    gray = cv2.normalize(gray, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX)
    return gray


# ===================== MOVE WITH TIMER (BODY) =====================

def move_with_timer(direction, duration, speed=0.5):
    """
    Di chuyển theo timer trong BODY (tiến, lùi, trái, phải).
    Hàm tiện ích, gọi qua get_controller() bên trong.
    """
    from drone_control import get_controller   # import lazy để tránh vòng lặp

    ctrl = get_controller()

    vx, vy = 0, 0
    if direction == 'forward':
        vx = speed
    elif direction == 'backward':
        vx = -speed
    elif direction == 'left':
        vy = -speed
    elif direction == 'right':
        vy = speed
    else:
        return

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            ctrl.send_local_ned_velocity(vx, vy, 0)
            time.sleep(0.1)
    finally:
        ctrl.send_local_ned_velocity(0, 0, 0)


# ===================== DRONE CONTROLLER =====================

class DroneController:
    def __init__(self, connection_str='udp:100.69.194.50:5000', takeoff_height=4):
        """
        Create DroneController and connect to vehicle.
        """
        self.connection_str = connection_str
        print(" Connecting to vehicle on", connection_str)

        try:
            self.vehicle = connect(
                connection_str,
                baud=115200,
                wait_ready=True,
                timeout=120
            )
            print(" Vehicle connected successfully")
        except Exception as e:
            print(f" Failed to connect to vehicle: {e}")
            self.vehicle = None

        # ===== TELEMETRY BUFFER =====
        self._telemetry_lock = threading.Lock()
        self.latest_telemetry = {
            'lat': None,
            'lon': None,
            'alt': None,
            'mode': None,
            'velocity': 0.0,
            'connected': bool(self.vehicle)
        }

        if self.vehicle:
            try:
                # Landing parameters
                self.vehicle.parameters['PLND_ENABLED'] = 1
                self.vehicle.parameters['PLND_TYPE'] = 1
                self.vehicle.parameters['PLND_EST_TYPE'] = 0
                self.vehicle.parameters['LAND_SPEED'] = 30

                # Đăng ký listeners
                self.vehicle.add_attribute_listener(
                    'location.global_frame', self._location_listener
                )
                self.vehicle.add_attribute_listener(
                    'location.global_relative_frame', self._rel_location_listener
                )
                self.vehicle.add_attribute_listener(
                    'velocity', self._velocity_listener
                )
                self.vehicle.add_attribute_listener(
                    'mode', self._mode_listener
                )

                print("Landing parameters & telemetry listeners set successfully")
            except Exception as e:
                print(" Failed to set some landing parameters/listeners:", e)

        self.takeoff_height = takeoff_height
        self.flown_path = []

        # ArUco
        self.aruco_thread = None
        self.aruco_running = False
        self.find_aruco = list(range(33))  # 0–32

        # Zone parameters (full frame)
        self.zone_center_offset = [0, 0]
        self.zone_width = 1280
        self.zone_height = 720

    # -------- Telemetry listeners --------
    def _location_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            lat = float(value.lat) if value.lat is not None else None
            lon = float(value.lon) if value.lon is not None else None

            with self._telemetry_lock:
                self.latest_telemetry['lat'] = lat
                self.latest_telemetry['lon'] = lon
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Location listener error:", e)

    def _rel_location_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            alt = float(value.alt) if value.alt is not None else None

            with self._telemetry_lock:
                self.latest_telemetry['alt'] = alt
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Relative location listener error:", e)

    def _velocity_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            vx, vy, vz = value
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            with self._telemetry_lock:
                self.latest_telemetry['velocity'] = speed
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Velocity listener error:", e)

    def _mode_listener(self, vehicle, attr_name, value):
        try:
            mode_name = value.name if value is not None else None
            with self._telemetry_lock:
                self.latest_telemetry['mode'] = mode_name
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Mode listener error:", e)

    # ---------------- CAMERA WRAPPER ----------------
    def start_image_stream(self, topic_name=None):
        try:
            start_camera()
            print(" Started camera stream")
        except Exception as e:
            print(" Failed to start camera:", e)

    def stop_image_stream(self):
        try:
            stop_camera()
            print(" Stopped camera stream")
        except Exception as e:
            print(" Failed to stop camera:", e)

    # ---------------- ARUCO CONFIG ----------------
    def set_find_aruco(self, ids):
        if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
            self.find_aruco = ids
            print(f" Updated ArUco IDs to detect: {ids}")
        else:
            raise ValueError(" Invalid ArUco IDs. Must be list of integers.")

    def send_aruco_marker_to_server(self, markers):
        """
        markers: dict {marker_id(str): {'lat': float, 'lon': float, 'selected': bool}}
        """
        if not markers:
            return
        try:
            payload = {'markers': markers}
            print(f"Sending ArUco markers to server: {payload}")
            response = requests.post(
                'http://127.0.0.1:5000/update_aruco_markers',
                json=payload,
                timeout=2
            )
            if response.status_code == 200:
                print(" Successfully sent ArUco markers to server")
            else:
                print(f"Failed to send ArUco markers, status code: {response.status_code}")
        except Exception as e:
            print(f" Error sending ArUco markers to server: {e}")

    # ---------------- MAVLINK HELPERS ----------------
    def send_local_ned_velocity(self, vx, vy, vz):
        """
        Gửi velocity trong BODY_OFFSET_NED frame.
        vx, vy, vz: m/s (vx: forward, vy: right, vz: down)
        """
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,  # enable vx, vy, vz
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_speed(self, speed):
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"Set speed to {speed} m/s")

    # ---------------- COMPASS HEADING ----------------
    def set_fixed_heading(self, heading_deg, yaw_rate=10, relative=False):
        """
        Đặt heading (la bàn) cố định bằng MAV_CMD_CONDITION_YAW.
        Tự chọn chiều quay ngắn nhất từ heading hiện tại.
        """
        if not self.vehicle:
            return

        current = getattr(self.vehicle, 'heading', None)
        if current is None:
            direction = 1  # fallback: clockwise
        else:
            # diff trong [0,360)
            diff = (heading_deg - current + 360.0) % 360.0
            # <=180 quay CW, >180 quay CCW
            direction = 1 if diff <= 180.0 else -1

        is_relative = 1 if relative else 0

        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            float(heading_deg),
            float(yaw_rate),
            float(direction),
            float(is_relative),
            0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(
            f"Set fixed heading to {heading_deg}° "
            f"(direction={direction}, relative={bool(is_relative)})"
        )

    # ---------------- MOVE WITH COMPASS (TIMER) ----------------
    def move_with_compass_timer(self, heading_deg, duration, speed=1.0, yaw_rate=15.0):
        """
        Bay theo hướng compass cố định trong thời gian 'duration' (s),
        sử dụng velocity trong BODY frame (vx = speed).
        1) Ép mode GUIDED
        2) Quay về heading tuyệt đối = heading_deg
        3) Chờ heading ổn định
        4) Gửi vx = speed trong 'duration' giây
        """
        if not self.vehicle:
            print("No vehicle connected for move_with_compass_timer")
            return

        # Đảm bảo GUIDED
        if self.vehicle.mode.name != "GUIDED":
            print(f"Current mode is {self.vehicle.mode.name}, switching to GUIDED...")
            # self.vehicle.mode = VehicleMode("GUIDED")
            while self.vehicle.mode.name != "GUIDED":
                print(" Waiting for GUIDED mode...")
                time.sleep(0.5)

        # Set heading mong muốn
        self.set_fixed_heading(heading_deg, yaw_rate=yaw_rate, relative=False)

        # Chờ heading ổn định (tối đa 10s)
        max_wait = 10.0
        t0 = time.time()
        stable_count = 0

        while time.time() - t0 < max_wait:
            current_heading = getattr(self.vehicle, 'heading', None)
            if current_heading is None:
                time.sleep(0.1)
                continue

            diff = abs(current_heading - heading_deg)
            diff = min(diff, 360 - diff)  # wrap 0–360

            if diff < 2.0:  # sai số ±2°
                stable_count += 1
                if stable_count >= 5:  # ~0.5s
                    print(
                        f"Heading stable at ~{current_heading:.1f}° "
                        f"(target {heading_deg}°)"
                    )
                    break
            else:
                stable_count = 0

            time.sleep(0.1)

        print(
            f"[MOVE COMPASS] heading={heading_deg}°, "
            f"speed={speed} m/s, duration={duration} s"
        )

        # Bay thẳng trong 'duration' giây
        period = 0.1
        next_t = time.time()
        end_t = next_t + duration

        try:
            while time.time() < end_t:
                current_heading = self.vehicle.heading
                if abs(current_heading - heading_deg) > 5:
                    self.set_fixed_heading(heading_deg)

                self.send_local_ned_velocity(speed, 0, 0)
                next_t += period
                sleep_time = next_t - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            # Luôn đảm bảo dừng lệnh velocity
            self.send_local_ned_velocity(0, 0, 0)
            print("[MOVE COMPASS] Done.")

    # ---------------- COMPASS MISSION ----------------
    def run_compass_mission(self, steps,
                            takeoff_height=None,
                            auto_takeoff=True,
                            land_at_end=True,
                            yaw_rate=15.0):
        """
        Bay lần lượt các step theo compass:
        steps: list các dict:
          {
            "heading_deg": float,   # hướng la bàn 0–360 (0 = Bắc)
            "speed": float,         # m/s
            "duration": float       # giây
          }
        """
        if not self.vehicle:
            print("No vehicle connected for run_compass_mission")
            return

        if not steps or len(steps) == 0:
            print("No steps provided for compass mission")
            return

        h = takeoff_height if takeoff_height is not None else self.takeoff_height

        if auto_takeoff:
            print(f"[COMPASS MISSION] Takeoff to {h} m")
            self.arm_and_takeoff(h)
            time.sleep(1.0)
            # Nếu muốn bật ArUco trong mission thì dùng:
            # self.start_aruco_processing()

        for idx, step in enumerate(steps, start=1):
            try:
                heading = float(step.get('heading_deg', 0.0))
                speed = float(step.get('speed', 1.0))
                duration = float(step.get('duration', 0.0))
            except Exception as e:
                print(f"[COMPASS MISSION] Invalid step {idx}: {step}, error={e}")
                continue

            if duration <= 0:
                print(f"[COMPASS MISSION] Skip step {idx} (duration <= 0)")
                continue

            print(
                f"[COMPASS MISSION] Step {idx}: "
                f"heading={heading}°, speed={speed} m/s, duration={duration} s"
            )

            self.move_with_compass_timer(
                heading_deg=heading,
                duration=duration,
                speed=speed,
                yaw_rate=yaw_rate
            )
            time.sleep(1.0)

        if land_at_end:
            print("[COMPASS MISSION] All steps done, starting LAND...")
            try:
                self.vehicle.mode = VehicleMode("LAND")
                while self.vehicle.mode.name != "LAND":
                    print(" Waiting for LAND mode...")
                    time.sleep(1.0)

                while self.vehicle.armed:
                    print(" Waiting for disarm...")
                    time.sleep(1.0)

                print("[COMPASS MISSION] Landing complete. Mission finished.")
            except Exception as e:
                print("Error during landing in compass mission:", e)

    # ---------------- GOTO / WAYPOINTS ----------------
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=1.5, timeout=60, speed=0.7):
        """
        simple_goto với tolerance & timeout nới lỏng, có record vị trí.
        """
        if speed < 0.1 or speed > 5.0:
            print(f"Tốc độ {speed} m/s không hợp lệ, đặt về 0.7 m/s")
            speed = 0.7

        if not self.vehicle:
            return False

        distanceToTargetLocation = self.get_distance_meters(
            targetLocation,
            self.vehicle.location.global_relative_frame
        )
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_dist = distanceToTargetLocation
        start_time = time.time()

        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(
                targetLocation,
                self.vehicle.location.global_relative_frame
            )
            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.02)

        print("Timeout reaching waypoint, proceeding anyway")
        return False

    # ---------------- ARM / TAKEOFF ----------------
    def arm_drone(self):
        """
        Arm drone không cất cánh
        """
        if not self.vehicle:
            return False

        while self.vehicle.mode.name != 'GUIDED':
            print(' Waiting for GUIDED mode...')
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)

        print("Drone is armed and ready")
        return True

    def arm_and_takeoff(self, targetHeight):
        if not self.vehicle:
            return

        while not self.vehicle.is_armable:
            print(' Waiting for vehicle to become armable')
            time.sleep(1)

        while self.vehicle.mode.name != 'GUIDED':
            print('Waiting for GUIDED...')
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)

        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f'📊 Altitude: {alt:.2f}' if alt else 'Altitude: 0.00')
            if alt and alt >= 0.95 * targetHeight:
                break
            time.sleep(1)

        print("Reached takeoff altitude")

    # ---------------- ARUCO PROCESSING ----------------
    def start_aruco_processing(self):
        """
        Start ArUco marker detection in separate thread
        """
        if self.aruco_running:
            print(" ArUco processing already running")
            return

        self.aruco_running = True
        self.aruco_thread = threading.Thread(
            target=self._aruco_processing_loop,
            daemon=True
        )
        self.aruco_thread.start()
        print("Started ArUco marker detection")

    def _aruco_processing_loop(self):
        global time_last

        print("ArUco processing started - ready to detect markers")

        while self.aruco_running:
            try:
                if time.time() - time_last < time_to_wait:
                    time.sleep(0.01)
                    continue

                time_last = time.time()

                frame_jpeg = get_lastest_frame()
                if frame_jpeg is None:
                    time.sleep(0.01)
                    continue

                nparr = np.frombuffer(frame_jpeg, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if cv_image is None:
                    continue

                detected_markers = self._process_frame_for_aruco(cv_image)

                if detected_markers:
                    try:
                        self.send_aruco_marker_to_server(detected_markers)
                    except Exception as e:
                        print("Error sending ArUco marker:", e)

            except Exception as e:
                print("ArUco processing error:", e)
                time.sleep(0.1)

    def _process_frame_for_aruco(self, cv_image):
        """
        Process a single frame for ArUco marker detection, only if within rectangular zone.
        Trả về dict: {marker_id(str): {'lat': float, 'lon': float, 'selected': bool}}
        """
        detected_markers = {}

        gray_img = preprocess_aruco_image(cv_image)

        corners, ids, rejected = aruco.detectMarkers(
            gray_img,
            aruco_dict,
            parameters=parameters
        )

        # Tâm khung hình (có offset zone)
        center_x = cv_image.shape[1] // 2 + self.zone_center_offset[0]
        center_y = cv_image.shape[0] // 2 + self.zone_center_offset[1]

        left = center_x - self.zone_width // 2
        right = center_x + self.zone_width // 2
        top = center_y - self.zone_height // 2
        bottom = center_y + self.zone_height // 2

        # Vẽ zone debug
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

        if ids is not None:
            ids_flat = ids.flatten()
            for idx, marker_id in enumerate(ids_flat):
                marker_id = int(marker_id)

                if marker_id < 0 or marker_id > 32:
                    continue

                is_selected = marker_id in self.find_aruco

                corner = corners[idx][0]
                marker_center_x = int(np.mean(corner[:, 0]))
                marker_center_y = int(np.mean(corner[:, 1]))

                if not (left <= marker_center_x <= right and
                        top <= marker_center_y <= bottom):
                    continue

                marker_size = 40  # cm
                try:
                    ret = aruco.estimatePoseSingleMarkers(
                        corners, marker_size,
                        cameraMatrix=np_camera_matrix,
                        distCoeffs=np_dist_coeff
                    )
                    rvecs, tvecs = ret[0], ret[1]
                    rvec = rvecs[idx][0, :]
                    tvec = tvecs[idx][0, :]
                    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                except Exception as e:
                    print(f"Error in estimatePoseSingleMarkers for ID {marker_id}: {e}")
                    continue

                marker_position = (
                    f'MARKER DETECTED - ID: {marker_id}, '
                    f'POS: x={x:.2f} y={y:.2f} z={z:.2f}, '
                    f'selected={is_selected}'
                )
                print(f"🎯 {marker_position}")

                # GPS ghi marker
                if self.vehicle:
                    try:
                        loc = self.vehicle.location.global_frame
                        if loc is None or loc.lat is None or loc.lon is None:
                            loc = self.vehicle.location.global_relative_frame

                        if loc and loc.lat and loc.lon:
                            lat = float(loc.lat)
                            lon = float(loc.lon)
                            detected_markers[str(marker_id)] = {
                                'lat': lat,
                                'lon': lon,
                                'selected': bool(is_selected)
                            }
                            print(
                                f" Recorded marker ID {marker_id} at "
                                f"lat={lat:.6f}, lon={lon:.6f}, selected={is_selected}"
                            )
                        else:
                            print(f" Invalid GPS coordinates for marker {marker_id}")
                    except Exception as e:
                        print(f"Error getting location for marker {marker_id}: {e}")

        return detected_markers

    def stop_aruco_processing(self):
        """
        Stop ArUco processing thread
        """
        self.aruco_running = False
        if self.aruco_thread:
            self.aruco_thread.join(timeout=2.0)
        print("Stopped ArUco processing")

    # ---------------- PATH UTILS / MISSION ----------------
    def interpolate_path(self, path, num_points=20):
        """
        Interpolate the recorded path to generate a smooth set of waypoints.
        """
        if not path or len(path) < 2:
            return path
        path = np.array(path)
        t = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_points)
        lat = np.interp(t_new, t, path[:, 0])
        lon = np.interp(t_new, t, path[:, 1])
        return [[lat[i], lon[i]] for i in range(num_points)]

    def fly_and_precision_land_with_waypoints(self, waypoints,
                                              takeoff_height=4,
                                              aruco_duration=30):
        """
        Fly to waypoints while detecting ArUco markers
        """
        if not self.vehicle:
            print(" No vehicle connected")
            return

        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        self.flown_path = []

        print("Arming and taking off")
        self.arm_and_takeoff(takeoff_height)
        time.sleep(1)

        self.start_aruco_processing()

        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, takeoff_height)
        print(f" Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly middle waypoints
        for i, wp in enumerate(waypoints[1:-1]):
            speed = wp.get('speed', 0.7)
            wp_loc = LocationGlobalRelative(wp['lat'], wp['lon'], takeoff_height)
            print(
                f"Flying to waypoint {i + 1}: {wp['lat']}, {wp['lon']} "
                f"at speed {speed} m/s"
            )
            self.goto(wp_loc, speed=speed)

        # Final goal
        goal_wp = waypoints[-1]
        speed = goal_wp.get('speed', 0.7)
        wp_target = LocationGlobalRelative(
            goal_wp['lat'], goal_wp['lon'], takeoff_height
        )
        print(
            f"Flying to final target {goal_wp['lat']}, {goal_wp['lon']} "
            f"at speed {speed} m/s"
        )
        self.goto(wp_target, speed=speed)

        self.stop_aruco_processing()

        print("Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")


# ===================== SINGLETON CONTROLLER =====================

_controller = None


def get_controller(connection_str='udp:100.69.194.50:5000', takeoff_height=5):
    global _controller
    if _controller is None:
        _controller = DroneController(
            connection_str=connection_str,
            takeoff_height=takeoff_height
        )
    return _controller


# ===================== CAMERA CLEANUP =====================

def cleanup_camera():
    """Cleanup camera resources"""
    global _picamera2, _camera_running
    _camera_running = False
    if _picamera2:
        try:
            _picamera2.stop()
            _picamera2 = None
        except Exception as e:
            print("Error stopping camera:", e)


import atexit
atexit.register(cleanup_camera)
