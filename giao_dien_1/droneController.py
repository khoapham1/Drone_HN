import time
import math
import cv2
from dronekit import connect, LocationGlobalRelative, VehecleMode
from pymavlink import mavutil
from tien_xu-lu import preprocess_aruco_image

time_to_wait = 0.1
#### CAU_HINH_ARUCO ####
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6x6_250) #tra ve ma tran bit cua 250 marker 6x6
parameters = cv2.aruco.DetectorParameters_create() # tham so dau vao de detect Aruco
# tham so cau hinh them
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX #Goc marker bam chinh xac hon duong bien den trang, it rung va POSE on dinh

parameters.adaptiveThreshWinSizeMin = 3 # kich thuoc cua cua so toi thieu de phat hien marker
parameters.adaptiveThreshWinSizeMax = 33 # kich thuoc cua so toi da de
parameters.adaptiveThreshWinSizeStep = 10 # buoc tang kich thuoc cua so

parameters.minMarkerPerimeterRate = 0.02 # ti le giua chu vi nho nhat cua marker va kich thuoc hinh anh
parameters.maxMarkerPerimeterRate = 4.0 # ti le giua chu vi lon nhat cua marker va kich thuoc hinh anh

#Gioi han ty le canh de tranh meo
parameters.minCornerDistanceRate = 0.05 # ti le giua khoang cach nho nhat giua cac goc va chu vi cua marker
parameters.minDistanceToBorder = 0.05 # khoang cach toi thieu tu marker den bien
#tinh chinh threshold
parameters.adaptiveThreshConstant = 7 # hang so duoc tru di tu gia tri trung


# --- DroneControll class ---
class DroneControll:
    def __init__(self, connection_str=connection_str, takeoff_height=4):
        self.connection_str ='udp:100.69.194.50:5000'
        print("Connecting to vehicle on", connection_str)

        try:
            self.vehicle = connect(connection_str, baud=115200, wait_ready = True, timeout = 60)
            print("Vehicle connect successfully")
        except Exception as e:
            print(f"Failed to connect to vehicle: {e}")
            self.vehicle = None

        if self.vehicle:
            try:
                self.vehicle.parameters['PLND_ENABLED'] = 1
                self.vehicle.parameters['PLND_TYPE'] = 1  # ArUco-based precision landing
                self.vehicle.parameters['PLND_EST_TYPE'] = 0
                self.vehicle.parameters['LAND_SPEED'] = 20
                print("Landing parameters set successfully")
            except Exception as e:
                print(" Failed to set some landing parameters:", e)
        
        self.takeoff_height = takeoff_height
        self.flow_path = []
        self.aruco_thread = None
        self.aruco_running = False
        #danh sach ArUco IDx can detect
        self.find_aruco = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    ########## CAMERA ############
    def start_image_stream(self, topic_name=None):
        try:
            start_camera()
            print("Bat dau stream Camera")
        except Exception as e:
            print("Failed to start camera", e)
    def stop_image_stream(self):
        try:
            stop_camera()
            print("Stop stream camera")
        except Exception as e:
            print("Failed stop camera", e)
    ########## SEND ARUCO ############
    def set_find_aruco(self, idx):
        if isinstance(ids, list) and all(isinstance(id, int) for id in ids):
            self.find_aruco = ids
            print(f"Updated ArUco IDs to detect: {idx}")
        else:
            raise ValueError("Hay nhap lai aruco khong trong danh sach can detect")
    
    def send_aruco_marker_to_server(self, markers):
        if not markers:
            return
        try:
            payload = {'marker': markers}
            print(f"Gui ArUco marker toi Server: {payload}")
            response = request.post(
                'http://127.0.0.1:5000/update_aruco_marker',
                json = payload,
                timeout = 2
            )
            if response.status_code == 200:
                print("Successfully sent ArUco markers to server")
            else:
                print(f"Loi gui den server ArUco marker: {response.status_code0}")
        except Exception as e:
            print(f"Error sending ArUco marker{e}")

    def send_local_ned_velocity(self, vx, vy, vz):
        if not self.vehicle:
            return
        msg = self.vehicle.message_facory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
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
        msg = self.vehicle.message_facory.command_long_encode(
            0,0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"Set speed to {speed} m/s")

    def get_distance_meter(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLat * dLat) + (dLon * dLon)) *1.11319e5

    def goto(self, targetLocation, tolerance=1.7, timeout=60,currentLocation):
        if speed < 0.1 or speed > 5.0:
            print(f"Toc do hien tai {speed} m/s khong hop le")
            speed = 0.7
        if not self.vehicle:
            return False
        khoach_cachTarget_location = self.get_distance_meter(targetLocation, self.vehicle.location.global_relative_frame)
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_dist = khoach_cachTarget_location
        start_time = time.time()
        while self.vehicle.mode.name == "GUIDED" or time.time() - start_time < timeout:
            vi_tri_hien_tai = self.get_distance_meter(targetLocation, self.vehicle.location.global_relative_frame)
            if vi_tri_hien_tai < max(tolerance, start_dist*0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.01)
        print("Time out reaching waypoint, proceeding anyway")
        return False
    def arm_and_takeoff(self, takeoff_height):
        if not self.vehicle:
            return
        while not self.vehicle.is_armable:
            print("Waiting for Armed....")
            time.sleep(1)

        while self.vehicle.mode != "GUIDED":
            print("Waiting for change mode GUIED")
            time.sleep(1)
        
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("Waiting for armble")
            time.sleep(1)

        self.vehicle.simple_takeoff(takeoff_height)

        while True:
            alt = self.vehicle.location.global_relative_frame.alt

            print(f'Altitude: {alt:.2f}' if alt else 'Altitude: 0.0' )

            if alt and alt >= 0.95 * takeoff_height:
                break
            time.sleep(1)
        print("Reached takeoff altitude")
        return None
### --- ARUCO PROCESSING THREAD ---
    def start_aruco_processing(self):
        if self.aruco_running:
            print("ArUco processing already running")
            return
        self.aruco_running = True
        self.aruco_thread = threading.Thread(target=self.aruco_processing_loop, daemon=True)
        self.aruco_thread.start()
        print("Started ArUco detect")

    def aruco_processing_loop(self):
        global time_last
        print("Start ArUco processing loop")

        while self.aruco_running:
            try:
                if time.time() - time_last < time_to_wait:
                    time.sleep(0.01)
                    continue

                time_last = time.time()

                #lay frame tu camera
                frame_ipeg = get_lastest_frame()
                if frame_jpeg is None:
                    time.sleep(0.01)
                    continue

                nparr = np.from
                cv_image = cv2.imencode(nparr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    continue

                detected_markers = self.preprocess_frame_for_aruco(cv_image)

                #Gui ArUco den server
                if detected_markers:
                    try:
                        self.send_aruco_marker_to_server(detected_markers)
                    except Exception as e:
                        print(f"Error sending ArUco markers to server: {e}")
            except Exception as e:
                print(f"Error in ArUco processing loop: {e}")
                time.sleep(0.1)

    def preprocess_frame_for_aruco(self, cv_image):
        """
        Process tung frame cho ArUco marker detection.
        Tra ve dict: {marker_id(str): {'lat': float, 'lon': float, 'speed': float}}
        """
        #tao 1 mang luu tru marker duoc phat hien
        detected_marker = {}

        # Tien xu ly anh cho ArUco
        gray_img = preprocess_aruco_image(cv_image) # from tien_xu-lu.py

        corners, ids, rejected = cv2.aruco.detected_marker(
            gray_img,
            aruco_dict,
            parameters=parameters
        )
        if ids is not None:
            ids_flat = ids.flatten()
            for idx, marker_id in enumerate(ids_flat):
                marker_id = int(marker_id)

                if marker_id < 0 or marker_id > 32:
                    continue
                
                is_selected = marker_id in self.find_aruco

                corner = corners[idx][0]
                marker_size = 40 # cm
                try:
                    ret = cv2.aruco.estimatePoseSingleMarkers(
                        corners, marker_size,
                        cameraMatrix=np_camera_matrix,
                        distCoeffs=np_dist_coeff                        
                    )
                    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                    #x, y, z = tvec[0], tvec[1], tvec[2]
                    x = '{:.2f}'.format(tvec[0])
                    y = '{:.2f}'.format(tvec[1])
                    z = '{:.2f}'.format(tvec[2])
                ### --- tinh toan goc lech tu tam drone so voi ArUco marker  ---
                    x_sum = 0
                    y_sum = 0

                    x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
                    y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]

                    # Calculate center coordinates
                    x_avg = x_sum*0.25
                    y_avg = y_sum*0.25
                    
                    x_ang = (x_avg - horizontal_res*0.5) * (horizontal_fov/horizontal_res)
                    y_ang = (y_avg - vertical_res*0.5) * (vertical_fov/vertical_res)

                except Exception as e:
                    print(f"Error estimating pose for marker {marker_id}: {e}")
                    continue

                marker_position = (
                    f'MARKER DETECTED ID: {marker_id},'
                    f'POS: x={x:.2f}, y={y:.2f}, z={z:.2f}'
                    f'Selected: {is_selected}'
                )
                print(f'{marker_position}')

                color = (0, 255, 0) if is_selected else (0, 0, 255) # xanh la or do

                # ve khung marker
                try:
                    cv2.aruco.drawDetectedMarkers(cv_image, 
                    [corners[idx]], 
                    np.array([[marker_id]]), 
                    borderColor=color
                    )
                except Exception as e:
                    print(f"Error drawing marker {marker_id}: {e}")

                if self.vehicle:
                    try:
                        loc = self.vehicle.global_frame
                        if loc is None or loc.lat is None or loc.lon is None:
                            loc = self.vehicle.global_relative_frame
                        if loc and loc.lat and loc.lon:
                            lat = float(loc.lat)
                            lon = float(loc.lon)
                            detected_markers[str(marker_id)] = {
                                'lat': lat,
                                'lon': lon,
                                'selected' : bool(is_selected)
                            }
                            print(
                                f"Recorded marker ID {marker_id} at "
                                f"lat:{lat:.6f}, lon:{lon:.6f}, selected: {is_selected}"
                            )
                        else:
                            print(f" Invalid GPS coordinates for marker {marker_id}")
                    except Exception as e:
                        print(f"Error recording GPS for marker {marker_id}: {e}")
        return detected_markers

    def stop_aruco_processing(self):
        """
        Stop ArUco processing thread
        """
        self.aruco_running = False
        if self.aruco_thread:
            self.aruco_thread.join(timeout=2.0)
        print("Stopped ArUco processing")

    def fly_and_precision_land_with_waypoints(self, waypoints, takeoff_height=4, aruco_duration=30):
        """
        Fly to waypoints while detecting ArUco markers
        """
        
        if not self.vehicle:
            print(" No vehicle connected")
            return
            
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff from home
        print("Arming and taking off")
        self.arm_and_takeoff(takeoff_height)
        time.sleep(1)

        # Start ArUco detection
        self.start_aruco_processing()

        # Store home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, takeoff_height)
        print(f" Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly to waypoints [1:-1] (skip start, exclude goal)
        for i, wp in enumerate(waypoints[1:-1]):
            speed = wp.get('speed', 0.7)  # Lấy speed nếu có, else default
            wp_loc = LocationGlobalRelative(wp['lat'], wp['lon'], takeoff_height)
            print(f"Flying to waypoint {i+1}: {wp['lat']}, {wp['lon']} at speed {speed} m/s")
            self.goto(wp_loc, speed=speed)

        # Fly to final goal
        goal_wp = waypoints[-1]
        speed = goal_wp.get('speed', 0.7)
        wp_target = LocationGlobalRelative(goal_wp['lat'], goal_wp['lon'], takeoff_height)
        print(f"Flying to final target {goal_wp['lat']}, {goal_wp['lon']} at speed {speed} m/s")
        self.goto(wp_target, speed=speed)

        self.stop_aruco_processing()

        print("Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")

### ---- MOVE WITH TIMER ---- 
    def move_with_timer(direction, duration, speed):
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
        while time.time() - start_time < duration:
            send_local_ned_velocity(vx, vy, 0)
            time.sleep(0.1)
        send_local_ned_velocity(0, 0, 0)
    

    



                
            
                


                








    
    
        


