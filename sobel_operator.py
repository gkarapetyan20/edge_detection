import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detector(image_path):
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(image , cv2.CV_64F , 1 ,0 , ksize=3)
    sobel_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    threshold = 50
    edges = magnitude > threshold

    return edges

