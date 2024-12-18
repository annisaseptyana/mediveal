#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2 as cv
import json
from collections import defaultdict
import math

# Rectangle Segmentation
# Input : numpy of Image
# Output : List of multiple numpy Image
def rectangle_segmentation(input):
    img = input
    shape = img.shape
    
    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Thresholding
    ret, thresh = cv.threshold(gray, 110, 255, cv.THRESH_BINARY_INV)
    
    # Median Filter
    median = cv.medianBlur(thresh, 13)
    
    # Contour Extraction
    contours, hierarchy = cv.findContours(
        median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )[-2:]
    
    # Rectangle Detection
    # By Calculating Area and Perimeter Ratio from calculated contour
    ratio = np.zeros((len(contours), 1), np.float)
    rectangleContours = []
    
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        try:
            ratio = 16 * area / (perimeter * perimeter)
        except:
            ratio = 0
        if ratio < 1.05 and ratio > 0.95 and area > 1000:
        rectangleContours.append(c)
        
    # Sort Rectangle Image
    contourArray = rectangleContours
    
    # Get Rectangle max contour value
    maxVal = []

    for i in contourArray:
        m = np.amax(i, axis=0)
        maxVal.append(m)
    
    maxVal = np.array(maxVal)

    # Sorting based on y value
    max_indices_y = np.argsort(maxVal[:, 0, 1])
    
    max_sorted_indices_xy = []
    
    # Sorting each row based on x value
    # Row 1 Sorting
    startIndices = 0
    length = 2
    
    max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
    max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0, 0]) + startIndices]
    max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())
    
    # Row 2 Sorting
    startIndices = 2
    length = 4
    
    max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
    max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0, 0]) + startIndices]
    max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())
    
    # Row 3 Sorting
    startIndices = 6
    length = 4
    
    max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
    max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0, 0]) + startIndices]
    max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())
 
    # Row 4 Sorting
    startIndices = 10
    length = 2
 
    max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
    max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0, 0]) + startIndices]
    max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())

    contourArray_sorted = []
    for i in max_sorted_indices_xy:
        contourArray_sorted.append(contourArray[i])

    rectangleImage = []

    # Extract Rectangle Image
    for i, c in enumerate(contourArray_sorted):
        rect = cv.boundingRect(c)
        x, y, w, h = rect
        cropped = img[y : y + h, x : x + w]
        rectangleImage.append(cropped)

    return rectangleImage

# Circle Segmentation
# Input : numpy of Image
# Output : List of multiple numpy Image
def circle_segmentation(input):
    img = input
    
    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Thresholding
    ret, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
 
    # Canny Edge Detector
    blur = cv.GaussianBlur(thresh, (11, 11), 4)
    edges = cv.Canny(blur, 190, 200)
    edges = cv.dilate(edges, None, iterations=3)
    edges = cv.erode(edges, None, iterations=2)
 # Circle Hough Transform
    circles = cv.HoughCircles(
        image=edges,
        method=cv.HOUGH_GRADIENT,
        dp=1,
        minDist=190,
        param1=50,
        param2=10,
        minRadius=28,
        maxRadius=42,
    )

    hough = np.zeros((edges.shape[0], edges.shape[1]), np.uint8)
    circles = np.uint16(np.around(circles))
 
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(hough, (i[0], i[1]), i[2], (255, 255, 255), 2)
        # draw the center of the circle
        cv.circle(hough, (i[0], i[1]), 2, (0, 0, 255), 3)
        
# Labeling
    height, width = hough.shape
    
    mask = np.zeros((height + 2, width + 2), np.uint8)
    flags = 4
    flags |= cv.FLOODFILL_MASK_ONLY
    flags |= 255 << 8

    cv.floodFill(
        image=hough,
        mask=mask,
        seedPoint=(int(height / 2), int(width / 2)),
        newVal=(255, 0, 0),
        # loDiff=,
        # upDiff=(10,) * 3,
        flags=flags,
    )

    mask = ~mask

    ret, markers = cv.connectedComponents(mask)

    markers = np.uint8(markers)
    markers = markers[1:-1, 1:-1]

    # Masking
    masked_crop = []
    masked_endpoint = []

    i = 2
    while i <= markers.max():
        mask = np.uint8(markers == i)
        mask_position = np.where(mask == 1)
        startPoint = (mask_position[0].min(), mask_position[1].min())
        endPoint = (mask_position[0].max(), mask_position[1].max())
        
    masked = cv.bitwise_and(img, img, mask=mask)
    masked_crop_n = masked[startPoint[0] : endPoint[0], startPoint[1] : endPoint[1]]
    
    masked_crop.append(masked_crop_n)
    masked_endpoint.append(endPoint)

    i = i + 1

    # Sorting Circle Image

    maxVal = masked_endpoint
    maxVal = np.array(maxVal)

    # Sorting based on y value
    max_indices_y = np.argsort(maxVal[:, 1])

    max_sorted_indices_xy = []

    # Sorting each row based on x value
    # Row 1 Sorting
    startIndices = 0
    length = 2

     max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
     max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0]) + startIndices]
     max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())

    # Row 2 Sorting
    startIndices = 2
    length = 2

    max_row_n = maxVal[max_indices_y[startIndices : startIndices + length]]
    max_column_n_indices_x = max_indices_y[np.argsort(max_row_n[:, 0]) + startIndices]
    max_sorted_indices_xy.extend(max_column_n_indices_x.tolist())

    # Output Image
    circleImage = []
    for i in max_sorted_indices_xy:
        circleImage.append(masked_crop[i])
        i = i + 1
        
    return circleImage

# Color Measurement
# Input : numpy of Image
# Output : Tuple of (Dictionary of Color Value, List of joined color value)
def color_measurement(input):
    sampleSize = 20
    img = input
    shape = img.shape
    
    joinedOutput = []
    
    # Filtering
    median = cv.medianBlur(img, 13)
    
    sample = median[
        int(shape[0] / 2) - sampleSize : int(shape[0] / 2) + sampleSize,
        int(shape[1] / 2) - sampleSize : int(shape[1] / 2) + sampleSize,
    ]
    # RGB
    b = np.mean(sample[:, :, 0])
    g = np.mean(sample[:, :, 1])
    r = np.mean(sample[:, :, 2])
    
    output = {"bgr": [b, g, r]}
    joinedOutput.extend([b, g, r])

    # HSV
    hsvColor = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    h = np.mean(hsvColor[:, :, 0])
    s = np.mean(hsvColor[:, :, 1])
    v = np.mean(hsvColor[:, :, 2])
    output["hsv"] = [h, s, v]
    joinedOutput.extend([h, s, v])
    
    # Grayscale
    grayImage = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
    gray = np.mean(grayImage)
    output["gray"] = gray
    joinedOutput.extend([gray])
    
    # Output
    return output, joinedOutput
    
def main(imageDirectory):
    img = cv.imread(imageDirectory)
    output = {}
    outputJoined = []
    
    rectangleImage = rectangle_segmentation(img)
    
    for i, rectangle in enumerate(rectangleImage):
        output[i] = {}
        circleImage = circle_segmentation(rectangle)
        
        for j, circle in enumerate(circleImage):
            color, colorJoined = color_measurement(circle)
            
            output[i][j] = color
            
            indexColor = [i, j]
            indexColor.extend(colorJoined)
            outputJoined.append(indexColor)
            
            cv.imwrite("output/" + str(i) + "_" + str(j) + ".png",circle)
        cv.imwrite("output/" + str(i) + ".png", rectangle)
        
    with open("output.json", "w") as outfile:
        json.dump(output, outfile)
    np.savetxt("output.csv", outputJoined, delimiter=", ", fmt="% s")
    
main('filename.jpg')

