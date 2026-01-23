# server.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import time
from drone_control import get_controller, get_lastest_frame
import json
from planner import run_planner
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
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   logger=False, 
                   engineio_logger=False,
                   async_mode='threading')

# Create/connect controller
try:
    controller = get_controller(connection_str='/dev/ttyACM0', takeoff_height=3)
    print("✅ Drone controller initialized")
except Exception as e:
    print(f"❌ Failed to initialize drone controller: {e}")
    controller = None

# Mission variables
COMPASS_MISSION_STEP = {
    "heading_deg": 0.0,
    "speed": 1.0,
    "duration": 3.0
}
COMPASS_MISSION_STEPS = [COMPASS_MISSION_STEP.copy()]

# Recording variables
recording = False
recording_lock = Lock()
video_writer = None
recording_start_time = None

# Person detection storage
person_detections = {}
person_detections_lock = Lock()

# Start image stream
try:
    if controller:
        controller.start_image_stream()
        print("✅ Camera stream started")
    else:
        from drone_control import start_camera
        start_camera()
        print("✅ Camera started")
except Exception as e:
    print(f"⚠️ Failed to start camera: {e}")

def get_network_info():
    """Get network information"""
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
    """Check if port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result == 0
    except:
        return False

def cleanup():
    """Cleanup function"""
    print("🧹 Cleaning up resources...")
    try:
        if controller:
            controller.stop_person_detection()
            controller.stop_image_stream()
        from drone_control import cleanup as drone_cleanup
        drone_cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print('\n🛑 Shutting down server...')
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def mjpeg_generator():
    """Generate MJPEG stream with person detection overlay"""
    global recording, video_writer
    
    while True:
        try:
            frame_jpeg = get_lastest_frame()
            if frame_jpeg is None:
                placeholder = cv2.imencode('.jpg', np.zeros((1,1,3), np.uint8))[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.1)
                continue
            
            nparr = np.frombuffer(frame_jpeg, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Get person detections from controller
                if controller:
                    detections = controller.detected_persons
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        confidence = det['confidence']
                        
                        # Draw bounding box
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw confidence
                        cv2.putText(cv_image, 
                                   f"Person: {confidence:.2f}",
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.6, (0, 255, 0), 2)
                        
                        # Draw center point
                        center_x = det['center_x']
                        center_y = det['center_y']
                        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw detection count
                count = len(detections) if controller and hasattr(controller, 'detected_persons') else 0
                cv2.putText(cv_image, 
                           f"Persons Detected: {count}",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, (0, 255, 255), 2)
                
                # Record if enabled
                with recording_lock:
                    if recording and video_writer is not None:
                        try:
                            resized_frame = cv2.resize(cv_image, (1280, 720))
                            video_writer.write(resized_frame)
                        except Exception as e:
                            print(f"Error writing frame: {e}")
                
                # Encode to JPEG
                ret, jpeg = cv2.imencode('.jpg', cv_image, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 90,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ])
                if ret:
                    frame_jpeg = jpeg.tobytes()
        
        except Exception as e:
            print(f"Error in MJPEG generator: {e}")
            pass
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_person_detection', methods=['POST'])
def start_person_detection():
    try:
        if controller:
            controller.start_person_detection()
            return jsonify({'status': 'success', 'message': 'Person detection started'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_person_detection', methods=['POST'])
def stop_person_detection():
    try:
        if controller:
            controller.stop_person_detection()
            return jsonify({'status': 'success', 'message': 'Person detection stopped'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_person_detection', methods=['POST'])
def update_person_detection():
    """Receive person detection results from drone"""
    try:
        data = request.get_json(silent=True) or {}
        
        with person_detections_lock:
            detection_id = f"det_{int(time.time() * 1000)}"
            person_detections[detection_id] = {
                'lat': data.get('lat'),
                'lon': data.get('lon'),
                'alt': data.get('alt', 0),
                'timestamp': data.get('timestamp', time.time()),
                'count': data.get('count', 0),
                'detections': data.get('detections', [])
            }
            
            # Keep only last 100 detections
            if len(person_detections) > 100:
                oldest_key = min(person_detections.keys())
                del person_detections[oldest_key]
        
        # Send to frontend
        socketio.emit('person_detection_update', {
            'lat': data.get('lat'),
            'lon': data.get('lon'),
            'alt': data.get('alt', 0),
            'timestamp': data.get('timestamp', time.time()),
            'count': data.get('count', 0)
        })
        
        print(f"📍 Person detection: {data.get('count', 0)} persons at lat={data.get('lat', 0):.6f}, lon={data.get('lon', 0):.6f}")
        
        return jsonify({'status': 'success', 'count': data.get('count', 0)})
        
    except Exception as e:
        print(f"Error updating person detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_person_detections', methods=['GET'])
def get_person_detections():
    """Get all person detections"""
    with person_detections_lock:
        return jsonify(list(person_detections.values()))

@app.route('/clear_person_detections', methods=['POST'])
def clear_person_detections():
    """Clear all person detections"""
    with person_detections_lock:
        person_detections.clear()
    socketio.emit('person_detections_cleared', {})
    return jsonify({'status': 'cleared'})

# Recording endpoints
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, video_writer, recording_start_time
    
    with recording_lock:
        if not recording:
            try:
                if not os.path.exists('recordings'):
                    os.makedirs('recordings')
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/flight_{timestamp}.avi"
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
                
                recording = True
                recording_start_time = time.time()
                
                print(f"🎥 Started recording: {filename}")
                socketio.emit('recording_status', {'status': 'started', 'filename': filename})
                
                return jsonify({'status': 'success', 'filename': filename})
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
                
                print(f"🛑 Stopped recording, duration: {duration:.2f}s")
                socketio.emit('recording_status', {'status': 'stopped', 'duration': duration})
                
                return jsonify({'status': 'success', 'duration': duration})
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

# Mission endpoints
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
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.get_json(silent=True) or {}
        station_name = payload.get('station', 'station1')
        
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)

        if station_name not in data:
            return jsonify({'error': f'Station "{station_name}" not found'}), 400

        v = controller.vehicle
        start = {"lat": v.location.global_frame.lat, "lon": v.location.global_frame.lon}
        waypoints = [start]
        
        for point in data[station_name]:
            wp = {"lat": point['lat'], "lon": point['lon']}
            if 'speed' in point:
                wp['speed'] = point['speed']
            waypoints.append(wp)
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        
        # Start mission in thread
        def mission():
            try:
                socketio.emit('mission_status', {'status': 'starting'})
                # Simple mission - fly to each waypoint
                for wp in waypoints[1:]:
                    target = LocationGlobalRelative(wp['lat'], wp['lon'], controller.takeoff_height)
                    speed = wp.get('speed', 0.7)
                    controller.goto(target, speed=speed)
                socketio.emit('mission_status', {'status': 'completed'})
            except Exception as e:
                socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
        
        threading.Thread(target=mission, daemon=True).start()
        
        return jsonify({'status': 'mission_started', 'station': station_name})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/takeoff', methods=['POST'])
def takeoff():
    try:
        if controller:
            height = request.json.get('height', 4)
            controller.arm_and_takeoff(height)
            return jsonify({'status': 'success', 'message': f'Takeoff to {height}m'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/land', methods=['POST'])
def land():
    try:
        if controller:
            controller.land()
            return jsonify({'status': 'success', 'message': 'Landing initiated'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
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

@app.route('/return_home', methods=['POST'])
def return_home():
    try:
        if not controller or not controller.vehicle:
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
                controller.fly_and_precision_land_with_waypoints(waypoints, takeoff_height=3)          
                socketio.emit('mission_status', {'status': 'completed'})
        except Exception as e:
            socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
    t = threading.Thread(target=mission, daemon=True)
    t.start()
    return t


# Compass mission endpoints
@app.route('/set_compass_mission', methods=['POST'])
def set_compass_mission():
    global COMPASS_MISSION_STEPS
    try:
        data = request.get_json(silent=True) or {}
        steps = data.get('steps', [])
        
        if not isinstance(steps, list):
            return jsonify({'error': 'steps must be a list'}), 400
        
        normalized_steps = []
        for idx, s in enumerate(steps):
            try:
                heading = float(s.get('heading_deg', 0))
                speed = float(s.get('speed', 1.0))
                duration = float(s.get('duration', 1.0))
                normalized_steps.append({
                    'heading_deg': heading,
                    'speed': speed,
                    'duration': duration
                })
            except Exception:
                return jsonify({'error': f'invalid step at index {idx}'}), 400
        
        COMPASS_MISSION_STEPS = normalized_steps
        return jsonify({'status': 'success', 'steps': COMPASS_MISSION_STEPS})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# SocketIO events
@socketio.on('change_mode')
def handle_change_mode(data):
    mode = data.get('mode')
    if mode in ['LAND', 'GUIDED', 'RTL']:
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

# Telemetry background task
def telemetry_loop():
    """Send telemetry data to clients"""
    while True:
        try:
            if controller and controller.vehicle:
                with controller._telemetry_lock:
                    data = dict(controller.latest_telemetry)
                
                # Add person detection count
                data['person_count'] = len(controller.detected_persons) if hasattr(controller, 'detected_persons') else 0
                
                socketio.emit('telemetry', data)
            else:
                socketio.emit('telemetry', {'connected': False})
            
            socketio.sleep(0.05)  # ~20 Hz
            
        except Exception as e:
            print(f"Telemetry error: {e}")
            socketio.sleep(1)

if __name__ == '__main__':
    print("🚀 Starting Drone Control Server...")
    
    hostname, main_ip, all_ips = get_network_info()
    
    print("🔍 Network Information:")
    print(f"   Hostname: {hostname}")
    print(f"   Main IP: {main_ip}")
    print(f"   All IPs: {', '.join(all_ips)}")
    
    if check_port_availability(5000):
        print("✅ Port 5000 is available")
    else:
        print("⚠️ Port 5000 might be in use")
    
    print("📡 Starting telemetry background task...")
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
        print("\n🛑 Server stopped by user")
        cleanup()
    except Exception as e:
        print(f"❌ Server error: {e}")
        cleanup()