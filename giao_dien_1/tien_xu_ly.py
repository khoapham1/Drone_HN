import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2
import math
class tien_xu_ly:

    def preprocess_aruco_image(bgr):
        """
        Tien xu ly anh cho ArUco
        - Chong troi / thieu sanh xu ly bang CLAHE + auto gamma
        - giam nhieu nhungw giu bien marker
        Tra ve anhr GRAY unit8.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridsize=(8,8))#tuong phan cuc bo
        gray = clahe.apply(gray)

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
            gray = np.clip(gray_gamma * 255.0, 0, 255).astype(np.unit8)
        gray = cv2.bilateralFilter(gary, d=5, sigmaColor=75, sigmaSpace=75)

        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return gray