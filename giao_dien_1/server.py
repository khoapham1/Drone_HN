# server.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import time
from drone_control import get_controller, get_lastest_frame, preprocess_aruco_image, aruco_dict, parameters
import json
from planner import run_planner  # Import planner
from dronekit import VehicleMode, LocationGlobalRelative
import math
import cv2
import numpy as np
import signal
import sys
import atexit
import socket
import os
import datetime
from threading import Lock

app = Flask(__name__)
# Sửa CORS settings để cho phép kết nối từ mọi nguồn
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   logger=False, 
                   engineio_logger=False,
                   async_mode='threading')

# Create/connect controller singleton
try:
    controller = get_controller(connection_str='udp:100.69.194.50:5000', takeoff_height=4)
    print(" Drone controller initialized successfully")
except Exception as e:
    print(f"Failed to initialize drone controller: {e}")
    controller = None

prev_loc = None
distance_traveled = 0.0
COMPASS_MISSION_STEP = {
    "heading_deg": 0.0,  # 0 độ = Bắc
    "speed": 1.0,        # m/s
    "duration": 3.0      # giây
}
COMPASS_MISSION_STEPS = [COMPASS_MISSION_STEP.copy()]

# Record start time
recording = False
recording_lock = Lock()
video_writer = None
recording_start_time = None

# Thêm variables global (hoặc trong hàm nếu muốn)
zone_center_offset = [0, 0]  # Offset
zone_width = 1280
zone_height = 720

# Lưu trữ thông tin ArUco markers
aruco_markers = {}

# Start image streamer so we always have frame to serve
try:
    if controller:
        controller.start_image_stream()
        controller.start_aruco_processing()
        print(" Camera and ArUco processing started successfully")
    else:
        from drone_control import start_camera
        start_camera()
        print("Camera started successfully")
except Exception as e:
    print(" Warning: Failed to start image streamer:", e)

def get_network_info():
    """Lấy thông tin mạng chi tiết"""
    try:
        hostname = socket.gethostname()
        all_ips = []
        for interface in socket.getaddrinfo(hostname, None):
            ip = interface[4][0]
            if ip not in all_ips and not ip.startswith('127.'):
                all_ips.append(ip)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            main_ip = s.getsockname()[0]
            s.close()
        except:
            main_ip = all_ips[0] if all_ips else "0.0.0.0"
        return hostname, main_ip, all_ips
    except Exception as e:
        return "unknown", "0.0.0.0", []

def check_port_availability(port=5000):
    """Kiểm tra xem port có sẵn sàng không"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result == 0
    except:
        return False

def cleanup():
    """Cleanup function to be called on exit"""
    print(" Cleaning up resources...")
    try:
        if controller:
            controller.stop_aruco_processing()
            controller.stop_image_stream()
    except Exception as e:
        print("Error during cleanup:", e)

# Register cleanup function
atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n Shutting down server...')
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def mjpeg_generator():
    """Generator that MJPEG frames with ArUco marker detection overlay and recording capability."""
    global recording, video_writer
    
    while True:
        try:
            frame = get_lastest_frame()
            if frame is None:
                placeholder = cv2.imencode('.jpg', np.zeros((1,1,3), np.uint8))[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.1)
                continue
            
            nparr = np.frombuffer(frame, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Process ArUco markers on the BGR frame
                processed_frame = process_aruco_on_frame(cv_image)
                
                # Ghi frame vào video nếu đang recording
                with recording_lock:
                    if recording and video_writer is not None:
                        try:
                            # Resize frame để đảm bảo kích thước phù hợp
                            resized_frame = cv2.resize(processed_frame, (1280, 720))
                            video_writer.write(resized_frame)
                        except Exception as e:
                            print(f"Error writing frame to video: {e}")
                
                # Encode the RGB frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 90,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ])
                if ret:
                    frame = jpeg.tobytes()
        
        except Exception as e:
            print(f" Error processing frame for ArUco: {e}")
            pass
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

def process_aruco_on_frame(cv_image):
    try:
        global aruco_markers

        # Tiền xử lý giống bên drone_control
        gray = preprocess_aruco_image(cv_image)

        # Tâm khung hình có offset
        center_x = cv_image.shape[1] // 2 + zone_center_offset[0]
        center_y = cv_image.shape[0] // 2 + zone_center_offset[1]

        # Vùng zone
        left   = center_x - zone_width  // 2
        right  = center_x + zone_width  // 2
        top    = center_y - zone_height // 2
        bottom = center_y + zone_height // 2

        # Vẽ zone lên frame
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        detected_count = 0
        if ids is not None:
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id[0])

                # chỉ quan tâm 0–32
                if marker_id < 0 or marker_id > 32:
                    continue

                # xem marker này có nằm trong danh sách chọn không
                is_selected = False
                try:
                    if controller:
                        is_selected = marker_id in controller.find_aruco
                except Exception:
                    pass

                corner = corners[i][0]
                marker_center_x = int(np.mean(corner[:, 0]))
                marker_center_y = int(np.mean(corner[:, 1]))

                # Marker phải nằm trong zone
                if not (left <= marker_center_x <= right and top <= marker_center_y <= bottom):
                    continue

                color = (0, 255, 0) if is_selected else (0, 0, 255)

                try:
                    cv2.aruco.drawDetectedMarkers(
                        cv_image,
                        [corners[i]],
                        np.array([[marker_id]]),
                        borderColor=color
                    )
                except Exception as e:
                    print(f"Error drawing ArUco marker on frame: {e}")

                cv2.putText(
                    cv_image,
                    f"ID: {marker_id}",
                    (marker_center_x - 90, marker_center_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )

                cv2.line(cv_image, (marker_center_x, marker_center_y),
                         (center_x, center_y), color, 2)

                detected_count += 1

        cv2.putText(cv_image, f"ArUco Markers Detected: {detected_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 2)

    except Exception as e:
        print(f" Error in ArUco frame processing: {e}")

    return cv_image

def run_compass_mission_in_thread(steps, auto_takeoff=True, land_at_end=True):
    def worker():
        try:
            socketio.emit('mission_status', {
                'status': 'compass_mission_start',
                'steps': steps
            })
            if controller:
                controller.run_compass_mission(
                    steps,
                    takeoff_height=controller.takeoff_height,
                    auto_takeoff=auto_takeoff,
                    land_at_end=land_at_end,
                    yaw_rate=15.0
                )
            socketio.emit('mission_status', {'status': 'compass_mission_completed'})
        except Exception as e:
            socketio.emit('mission_status', {
                'status': 'error',
                'error': str(e)
            })
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


# --- ROUTES ---
@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_aruco_detection', methods=['POST'])
def start_aruco_detection():
    try:
        if controller:
            controller.start_aruco_processing()
            return jsonify({'status': 'success', 'message': 'ArUco detection started'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_aruco_detection', methods=['POST'])
def stop_aruco_detection():
    try:
        if controller:
            controller.stop_aruco_processing()
            return jsonify({'status': 'success', 'message': 'ArUco detection stopped'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, video_writer, recording_start_time
    
    with recording_lock:
        if not recording:
            try:
                # Tạo thư mục recordings nếu chưa tồn tại
                if not os.path.exists('recordings'):
                    os.makedirs('recordings')
                
                # Tạo filename với timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/flight_{timestamp}.avi"
                
                # Khởi tạo video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
                
                recording = True
                recording_start_time = time.time()
                
                print(f"Started recording: {filename}")
                socketio.emit('recording_status', {'status': 'started', 'filename': filename})
                
                return jsonify({'status': 'success', 'message': 'Recording started', 'filename': filename})
            except Exception as e:
                print(f"Error starting recording: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Recording already in progress'}), 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, video_writer, recording_start_time
    
    with recording_lock:
        if recording and video_writer:
            try:
                recording = False
                video_writer.release()
                video_writer = None
                
                duration = time.time() - recording_start_time
                recording_start_time = None
                
                print(f"Stopped recording, duration: {duration:.2f}s")
                socketio.emit('recording_status', {'status': 'stopped', 'duration': duration})
                
                return jsonify({'status': 'success', 'message': 'Recording stopped', 'duration': duration})
            except Exception as e:
                print(f"Error stopping recording: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'No recording in progress'}), 400

@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    try:
        if not os.path.exists('recordings'):
            return jsonify({'recordings': []})
        
        recordings = []
        for filename in os.listdir('recordings'):
            if filename.endswith('.avi'):
                filepath = os.path.join('recordings', filename)
                file_size = os.path.getsize(filepath)
                recordings.append({
                    'filename': filename,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        
        return jsonify({'recordings': recordings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_aruco_markers', methods=['POST'])
def update_aruco_markers():
    global aruco_markers
    try:
        data = request.get_json(silent=True) or {}
        markers = data.get('markers', {})
        
        aruco_markers.update(markers)
        
        socketio.emit('aruco_markers_update', {'markers': aruco_markers})
        
        print(f"Updated ArUco markers: {list(markers.keys())}")
        return jsonify({'status': 'success', 'markers_received': len(markers)})
    except Exception as e:
        print(f"Error updating ArUco markers: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_aruco_markers', methods=['GET'])
def get_aruco_markers():
    """Endpoint để lấy danh sách ArUco markers hiện tại"""
    return jsonify(aruco_markers)


@app.route('/set_aruco_ids', methods=['POST'])
def set_aruco_ids():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get('ids', [])
        if not isinstance(ids, list) or not all(isinstance(id, int) for id in ids):
            return jsonify({'error': 'ArUco IDs must be a list of integers'}), 400
        
        if controller:
            controller.set_find_aruco(ids)
            print(f"Updated find_aruco to: {ids}")
            return jsonify({'status': 'success', 'ids': ids})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        print(f" Error setting ArUco IDs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_gps_stations', methods=['GET'])
def get_gps_stations():
    try:
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_mission', methods=['POST'])
def start_mission():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.get_json(silent=True) or {}
        station_name = payload.get('station', 'station1')
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)

        if station_name not in data:
            return jsonify({'error': f'station "{station_name}" not found in JSON'}), 400

        v = controller.vehicle
        start = {"lat": v.location.global_frame.lat, "lon": v.location.global_frame.lon}
        waypoints = [start]
        for point in data[station_name]:
            wp = {"lat": point['lat'], "lon": point['lon']}
            if 'speed' in point:
                wp['speed'] = point['speed']
            waypoints.append(wp)
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'station': station_name, 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fly_selected', methods=['POST'])
def fly_selected():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.get_json(silent=True) or {}
        points = payload.get('points', [])
        
        if not points:
            return jsonify({'error': 'No points selected'}), 400
            
        v = controller.vehicle
        start = {"lat": v.location.global_frame.lat, "lon": v.location.global_frame.lon}  # dict
        waypoints = [start]
        for point in points:
            wp = {"lat": point['lat'], "lon": point['lon']}
            if 'speed' in point:
                wp['speed'] = point['speed']
            waypoints.append(wp)
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500.

# ============== Compass mission endpoints ================
@app.route('/move_with_compass', methods=['POST'])
def move_with_compass():
    """
    Trigger chế độ MOVE with compass:
    - Nếu payload rỗng -> dùng COMPASS_MISSION_STEP mặc định.
    - Nếu payload có heading_deg/speed/duration -> override.
    """
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500

        payload = request.get_json(silent=True) or {}

        heading_deg = float(payload.get('heading_deg', COMPASS_MISSION_STEP['heading_deg']))
        speed = float(payload.get('speed', COMPASS_MISSION_STEP['speed']))
        duration = float(payload.get('duration', COMPASS_MISSION_STEP['duration']))

        # Bắt đầu trong thread riêng để không block Flask
        run_move_with_compass_in_thread(heading_deg, speed, duration)

        return jsonify({
            'status': 'compass_move_started',
            'heading_deg': heading_deg,
            'speed': speed,
            'duration': duration
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/set_compass_mission', methods=['POST'])
def set_compass_mission():
    """
    Cập nhật danh sách các bước bay theo compass từ web.
    Body JSON:
    {
      "steps": [
        {"heading_deg": 0,   "speed": 1.0, "duration": 24},
        {"heading_deg": 90,  "speed": 1.0, "duration": 35},
        ...
      ]
    }
    """
    global COMPASS_MISSION_STEP, COMPASS_MISSION_STEPS

    try:
        data = request.get_json(silent=True) or {}
        steps = data.get('steps', [])

        if not isinstance(steps, list) or len(steps) == 0:
            return jsonify({'error': 'steps must be a non-empty list'}), 400

        normalized_steps = []
        for idx, s in enumerate(steps, start=1):
            try:
                heading = float(s.get('heading_deg'))
                speed = float(s.get('speed'))
                duration = float(s.get('duration'))
            except Exception:
                return jsonify({'error': f'invalid step at index {idx}'}), 400

            normalized_steps.append({
                'heading_deg': heading,
                'speed': speed,
                'duration': duration
            })

        # Lưu lại global để /run_compass_mission hoặc /move_with_compass dùng
        COMPASS_MISSION_STEPS = normalized_steps
        COMPASS_MISSION_STEP = normalized_steps[0]  # step đầu cho nút MOVE with compass

        return jsonify({
            'status': 'compass_mission_updated',
            'steps': COMPASS_MISSION_STEPS
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/run_compass_mission', methods=['POST'])
def run_compass_mission():
    """
    Chạy mission compass theo danh sách step đã set (hoặc gửi kèm).
    Body JSON (optional):
    {
      "steps": [...],          # nếu không gửi -> dùng COMPASS_MISSION_STEPS
      "auto_takeoff": true,
      "land_at_end": true
    }
    """
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500

        data = request.get_json(silent=True) or {}
        steps = data.get('steps') or COMPASS_MISSION_STEPS
        auto_takeoff = bool(data.get('auto_takeoff', True))
        land_at_end = bool(data.get('land_at_end', True))

        if not steps or len(steps) == 0:
            return jsonify({'error': 'No compass steps configured'}), 400

        run_compass_mission_in_thread(steps, auto_takeoff, land_at_end)

        return jsonify({
            'status': 'compass_mission_started',
            'steps': steps,
            'auto_takeoff': auto_takeoff,
            'land_at_end': land_at_end
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/return_home', methods=['POST'])
def return_home():
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        v = controller.vehicle
        home_lat = v.location.global_frame.lat
        home_lon = v.location.global_frame.lon
        home_alt = 3
        home_location = LocationGlobalRelative(home_lat, home_lon, home_alt)
        v.simple_goto(home_location)
        return jsonify({'status': 'returning_home', 'home': [home_lat, home_lon]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_mission_in_thread(waypoints):
    def mission():
        try:
            socketio.emit('mission_status', {'status': 'starting', 'waypoints': waypoints})
            if controller:
                controller.fly_and_precision_land_with_waypoints(waypoints, takeoff_height=4, aruco_duration=60)
            socketio.emit('mission_status', {'status': 'completed'})
        except Exception as e:
            socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
    t = threading.Thread(target=mission, daemon=True)
    t.start()
    return t

def run_move_with_compass_in_thread(heading_deg, speed, duration):
    def worker():
        try:
            socketio.emit('mission_status', {
                'status': 'compass_move_start',
                'heading_deg': heading_deg,
                'speed': speed,
                'duration': duration
            })
            if controller:
                controller.move_with_compass_timer(heading_deg, duration, speed)
            socketio.emit('mission_status', {'status': 'compass_move_completed'})
        except Exception as e:
            socketio.emit('mission_status', {
                'status': 'error',
                'error': str(e)
            })
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

def run_rectangle_compass_in_thread(speed=1.0):
    def worker():
        try:
            socketio.emit('mission_status', {
                'status': 'rectangle_start',
                'speed': speed
            })
            if controller:
                controller.fly_rectangle_compass(speed=speed)
            socketio.emit('mission_status', {'status': 'rectangle_completed'})
        except Exception as e:
            socketio.emit('mission_status', {
                'status': 'error',
                'error': str(e)
            })
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_aruco_markers', methods=['POST'])
def clear_aruco_markers():
    global aruco_markers
    aruco_markers = {}
    socketio.emit('aruco_markers_update', {'markers': aruco_markers})
    return jsonify({'status': 'cleared'})

@app.route('/fly', methods=['POST'])
def fly_route():
    global distance_traveled
    distance_traveled = 0.0
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.json
        lat = payload.get('lat')
        lon = payload.get('lon')
        if lat is None or lon is None:
            return jsonify({'error': 'missing lat/lon'}), 400
        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        goal = [lat, lon]
        planner_payload = {"start": start, "goal": goal}
        waypoints = run_planner(planner_payload)
        socketio.emit('planned_path', {'waypoints': waypoints})
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('change_mode')
def handle_change_mode(data):
    mode = data.get('mode')
    if mode in ['LAND', 'GUIDED']:
        try:
            if controller:
                controller.vehicle.mode = VehicleMode(mode)
                emit('mission_status', {'status': f'mode_changed_to_{mode}'})
            else:
                emit('mission_status', {'status': 'error', 'error': 'Controller not available'})
        except Exception as e:
            emit('mission_status', {'status': 'error', 'error': str(e)})
    else:
        emit('mission_status', {'status': 'invalid_mode'})

@app.route('/arm', methods=['POST'])
def arm_drone():
    try:
        if controller:
            success = controller.arm_drone()
            if success:
                return jsonify({'status': 'armed', 'message': 'Drone armed successfully'})
            else:
                return jsonify({'error': 'Failed to arm drone'}), 500
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def telemetry_loop():
    """Send telemetry data to clients periodically (dùng buffer từ DroneController)."""
    global distance_traveled

    while True:
        try:
            if controller and controller.vehicle and hasattr(controller, 'latest_telemetry'):
                with controller._telemetry_lock:
                    data = dict(controller.latest_telemetry)  # copy

                # Bổ sung distance_traveled vào data gửi ra UI
                data['distance_traveled'] = distance_traveled

                socketio.emit('telemetry', data)
            else:
                socketio.emit('telemetry', {'connected': False})

            # Dùng socketio.sleep để không block event loop
            socketio.sleep(0.05)   # ~20 Hz, có thể chỉnh 0.02 nếu muốn ~50 Hz

        except Exception as e:
            print(f" Telemetry error: {e}")
            socketio.sleep(1)


if __name__ == '__main__':
    print("Starting Drone Control Server...")
    
    hostname, main_ip, all_ips = get_network_info()
    
    print("🔍 Network Information:")
    print(f"   Hostname: {hostname}")
    print(f"   Main IP: {main_ip}")
    print(f"   All IPs: {', '.join(all_ips)}")
    
    if check_port_availability(5000):
        print("Port 5000 is available")
    else:
        print("Port 5000 might be in use, trying anyway...")
    
    print("Starting telemetry background task...")
    socketio.start_background_task(telemetry_loop)

    try:
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        cleanup()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup()