import cv2
import numpy as np


def prewit_edge_detection(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    gaussian_img = cv2.GaussianBlur(gray,(3,3),0)


    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    img_prewittx = cv2.filter2D(gaussian_img, -1, kernelx)
    img_prewitty = cv2.filter2D(gaussian_img, -1, kernely)

    return img_prewittx + img_prewitty