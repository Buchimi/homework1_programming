"""
CS 4391 Homework 1 Programming
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# main function
if __name__ == '__main__':
    # Step 1: read the crack box image with cv2.imread
    im = cv2.imread("cracker_box.jpg")
    

    # Step 2: use cv2.cvtColor to convert RGB image to gray scale image 
    #(replace with your code)
    gray_image = cv2.cvtColor(src=im, code=cv2.COLOR_BGR2GRAY)

    # Step 3: use central difference to compute image gradient on the gray scale image
    #(replace with your code)
    gradient_x = np.zeros_like(gray_image, dtype=np.float32)
    gradient_y = np.zeros_like(gray_image, dtype=np.float32)
    row, col = gray_image.shape
    
    for i in range(1, row -1):
        for j in range(1, col - 1):
            top = gray_image[i -1][j] if i > 0 else 0
            left = gray_image[i][j-1] if j > 0 else 0
            bottom = gray_image[i+1][j] if i < row - 1 else 0
            right = gray_image[i][j+1] if j < col-1 else 0
            gradient_x[i][ j] = np.divide(np.subtract(right, left, dtype=np.float32),2, dtype=np.float32)
            gradient_y[i, j] = np.divide(np.subtract(bottom, top, dtype=np.float32),2, dtype=np.float32)
    
    # show result with matplotlib (no need to change code below)
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(gray_image, cmap = 'gray')
    ax.set_title('Original image')

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(gradient_x, cmap = 'gray')
    ax.set_title('Gradient X')

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(gradient_y, cmap = 'gray')
    ax.set_title('Gradient Y')

    plt.show()
