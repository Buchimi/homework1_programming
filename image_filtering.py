"""
CS 4391 Homework 1 Programming
"""

import cv2
from matplotlib.dates import drange
import numpy as np
import matplotlib.pyplot as plt

# main function
if __name__ == '__main__':
    # Step 1: read the crack box image with cv2.imread
    im = cv2.imread("cracker_box.jpg")


    # Step 2: use cv2.cvtColor to convert RGB image to gray scale image 
    #(replace with your code)
    gray_image = cv2.cvtColor(src=im, code=cv2.COLOR_BGR2GRAY)


    # Step 3: define the filter kernel as described in the homework description as a numpy array
    #(replace with your code)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Step 4: filter the image with the kernel
    #(replace with your code)
    row, col = gray_image.shape
    output = np.zeros(gray_image.shape)
    for i in range(0, row ):
        for j in range(0, col):
            slice = gray_image[max(i-1, 0) :min(i+2, row), max(j - 1, 0): min(j+2, col)]
            data = slice * kernel[0 if i > 0 else 1: 3 if i < row - 1 else 2,
                                  0 if j > 0 else 1: 3 if j < col - 1 else 2]
            output[i, j] = np.sum(data, dtype=np.float64) #np.clip(np.sum(data), 0, 255)

            
    # show result with matplotlib (no need to change code below)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(gray_image, cmap = 'gray')
    ax.set_title('Original image')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(output, cmap = 'gray')
    ax.set_title('Filtered image')

    plt.show()
