#!/usr/bin/env python3
import cv2
import time

def test_camera_fps(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Test different settings
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (640, 480),
        (320, 240)
    ]
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Test with different FPS settings
        for fps in [30, 60, 90]:
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"\nTesting {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
            # Measure actual FPS
            num_frames = 100
            start = time.time()
            
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
            
            end = time.time()
            measured_fps = num_frames / (end - start)
            print(f"Measured FPS: {measured_fps:.1f}")
    
    cap.release()

if __name__ == "__main__":
    test_camera_fps(0)