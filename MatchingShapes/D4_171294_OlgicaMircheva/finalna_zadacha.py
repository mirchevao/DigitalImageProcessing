# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:55:28 2020

@author: Olgica
"""

import cv2
import glob
import numpy as np



path = glob.glob("database/*.jpg")
database = []

for image in path:
    element = cv2.imread(image, 1)
    database.append(element)

path = glob.glob("query/*.jpg")
query = []

for image in path:
    element = cv2.imread(image, 1)
    query.append(element)
    
allImages = []
retArray = []
for imageQuery in query:
    img1 = imageQuery.copy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(img1, 127, 255, 0, cv2.THRESH_BINARY)
    c1, h1 = cv2.findContours(th1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour1 = c1[0]
    for imageDatabase in database:
        img2 = imageDatabase.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, th2 = cv2.threshold(img2, 127, 255, 0, cv2.THRESH_BINARY)
        c2, h2 = cv2.findContours(th2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        controur2 = c2[0]
        ret2 = cv2.matchShapes(contour1,controur2,1,0.0)
        
        if(ret2 < 0.01):
            retArray.append(ret2)
            np.sort(retArray)
            allImages.append(imageDatabase)
        
        

for img in allImages:
    img = cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
