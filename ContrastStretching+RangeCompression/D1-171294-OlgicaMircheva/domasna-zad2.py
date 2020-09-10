# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:55:21 2020

@author: Olgica Mircheva
"""


import cv2
import numpy as np


c = int(input())
image = cv2.imread("presentation.png", cv2.COLOR_BGR2GRAY)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i,j]
        new_pixel = c * (np.log10(pixel+1))
        image[i,j] = new_pixel
        
        
cv2.imwrite('saved_zadacha2.png', image)
cv2.imshow("ex",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
        