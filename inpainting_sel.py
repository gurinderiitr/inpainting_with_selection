"""
============
Seam Carving
============

This program helps you select a region of an image -> crop out the selected part ->
fill in the void -> get the edited image with filled void.



This example demonstrates how images can be resized using seam carving [1]_.
Resizing to a new aspect ratio distorts image contents. Seam carving attempts
to resize *without* distortion, by removing regions of an image which are less
important. In this example we are using the Sobel filter to signify the
importance of each pixel.

.. [1] Shai Avidan and Ariel Shamir
       "Seam Carving for Content-Aware Image Resizing"
       http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf

"""
import sys
from skimage import data, draw
from skimage import transform, util
import numpy as np
from skimage import filters, color
from matplotlib import pyplot as plt
import cv2
from skimage.restoration import inpaint
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
processing = False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, processing
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        processing = True
        print(x,y)
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        #image_copy = image.copy()
        
        # draw a rectangle around the region of interest
        cv2.rectangle(clone, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", clone)
        inpainting(refPt);

def inpainting(poly):
    print("Processing")
    # Create mask with three defect regions: left, middle, right respectively
    global image
    mask = np.zeros(image.shape[:-1])

    mask[poly[0][0]:poly[1][0], poly[0][1]:poly[1][1]] = 1
    # mask[160:180, 70:155] = 1
    # mask[30:60, 170:195] = 1

    # Defect image over the same region in each color channel
    image_defect = image.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 1
    image = inpaint.inpaint_biharmonic(image, mask,
                                            multichannel=True)
    cv2.namedWindow("image_processed")
    cv2.imshow("image_processed", image)
    processing = False

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    if processing:
        cv2.imshow("image", clone)
    else:
        cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
