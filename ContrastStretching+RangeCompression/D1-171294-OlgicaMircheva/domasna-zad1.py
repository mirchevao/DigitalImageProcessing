# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:41:08 2020

@author: Olgica Mircheva
"""

import cv2
import numpy as np


img = cv2.imread("examplee.jpg", cv2.IMREAD_GRAYSCALE)

norm_img1 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

norm_img1 = np.clip(norm_img1, 0, 1)
norm_img1 = (255*norm_img1).astype(np.uint8)

cv2.imwrite("normalized.jpg",norm_img1)

cv2.imshow('original',img)
cv2.imshow('normalized',norm_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()