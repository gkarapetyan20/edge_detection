import cv2
import matplotlib.pyplot as plt
from sobel_operator import sobel_edge_detector
from prewit import prewit_edge_detection
from argparse import ArgumentParser 

def main():
    parser = ArgumentParser()
    parser.add_argument('--image' , help='image path')

    args = parser.parse_args()

    sobel_edge = sobel_edge_detector(args.image)
    prewit_edge = prewit_edge_detection(args.image)
    image = cv2.imread(args.image,cv2.COLOR_BGR2RGB)
    plt.subplot(131),plt.imshow(image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(sobel_edge,cmap = 'gray')
    plt.title('Sobel Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133) , plt.imshow(prewit_edge,cmap = 'gray')
    plt.title('Prewit Edge Image'), plt.xticks([]), plt.yticks([])
 
    plt.show()

if __name__ == "__main__":
    main()
