# Take in image, convert to grey and numpy array
# Will need to create Mask
# Will need to preform convolution using mask with binary classifier
# Display image 

# Testing theory to see if output is better when blurred as pre-processing

import cv2
import os
import numpy as np
from scipy.signal import convolve2d

PATH = "339.jpg"
img = cv2.imread(PATH)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, 3,3)

#Bj
SOBEL_ROW_MASK = np.array([[-1,0,1], 
                            [-2,0,2], 
                            [-1,0,1]])

#Bi
SOBEL_COLUMN_MASK = np.array([[-1,-2,1], 
                            [0,0,0], 
                            [1,2,1]])

#bJ
ROBERTS_ROW_MASK = np.array([[1,0] 
                            [0,-1]])


#bI
ROBERTS_COLUMN_MASK = np.array([[0,1] 
                            [-1,0]])


cv2.imshow("Mask", ROBERTS_ROW_MASK)

def Robers_Edge_Extraction(img):
    # Using the masks created above
    # Get a Row and Column Convolution

    x = 1
    ROW_CONVOLUTION = convolve2d(img, ROBERTS_ROW_MASK, mode='same')
    ROW_CONVOLUTION_PADDED = convolve2d(img, ROBERTS_ROW_MASK, mode='same', boundary='wrap')
