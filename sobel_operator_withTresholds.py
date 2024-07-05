import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def sobel_edge_detector(image, threshold):
    """
    Apply Sobel edge detection on an image and return the edges based on a threshold.
    
    Parameters:
    - image: np.ndarray, input grayscale image.
    - threshold: float, threshold value for edge detection.
    
    Returns:
    - edges: np.ndarray, binary image of edges.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = magnitude > threshold
    return edges

def plot_images(image_path, thresholds):
    """
    Plot the original image and Sobel edge detected images for different thresholds.
    
    Parameters:
    - image_path: str, path to the input image.
    - thresholds: list, list of threshold values.
    """
    original_image = cv2.imread(image_path)
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, len(thresholds) + 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    
    for i, threshold in enumerate(thresholds, 2):
        edges = sobel_edge_detector(grayscale_image, threshold)
        title = f'Sobel Edge Image, T = {threshold}' if i < len(thresholds) + 1 else f'Optimal Threshold T = {threshold}'
        plt.subplot(1, len(thresholds) + 1, i)
        plt.imshow(edges, cmap='gray')
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    plt.savefig('Output.jpg')
    plt.show()

def calculate_brightness(image):
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    return brightness

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', help='image path')
    args = parser.parse_args()
    img_brightness = calculate_brightness(args.image)

    thresholds = [50, 100, 150, 200, int(img_brightness)]

    plot_images(args.image, thresholds)

if __name__ == "__main__":
    main()
