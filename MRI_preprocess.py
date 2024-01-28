import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from IPython.display import Image
import imutils
import os

mri_dir = 'C:/Users/Jan/SGH/magisterka/dane/brain_tumor_mri/dane/'
docelowe_dir = 'C:/Users/Jan/SGH/magisterka/dane/brain_tumor_mri_cropped'

def crop_image(image, plot=False):

    #convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #blur
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #apply binary threshhold
    img_thresh = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)[1]

    #apply erosion
    img_thresh = cv2.erode(img_thresh, None, iterations=2)

    #dilate images
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    #find shapes or the contour of images
    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #grab contours
    contours = imutils.grab_contours(contours)

    #find biggest contour
    c = max(contours, key=cv2.contourArea)

    #extract contour positions
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    #generate new image
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    #plot
    if plot:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()

    return new_image

#crop training images and save it to the directory we previously created
glioma = mri_dir + 'glioma'
meningioma = mri_dir + 'meningioma'
no_tumor = mri_dir + 'notumor'
pituitary = mri_dir + 'pituitary'

j = 0
for i in tqdm(os.listdir(glioma)):
    path = os.path.join(glioma, i)
    img = cv2.imread(path)
    img = crop_image(img, plot=False)

    if img is not None:
        img = cv2.resize(img, (240, 240))
        save_path = docelowe_dir + '/glioma/' + str(j) + '.jpg'
        cv2.imwrite(save_path, img)
        j = j + 1

j = 0
for i in tqdm(os.listdir(meningioma)):
    path = os.path.join(meningioma, i)
    img = cv2.imread(path)
    img = crop_image(img, plot=False)

    if img is not None:
        img = cv2.resize(img, (240, 240))
        save_path =  docelowe_dir + '/meningioma/' + str(j) + '.jpg'
        cv2.imwrite(save_path, img)
        j = j + 1

j = 0
for i in tqdm(os.listdir(no_tumor)):
    path = os.path.join(no_tumor, i)
    img = cv2.imread(path)
    img = crop_image(img, plot=False)

    if img is not None:
        img = cv2.resize(img, (240, 240))
        save_path =  docelowe_dir + '/notumor/' + str(j) + '.jpg'
        cv2.imwrite(save_path, img)
        j = j + 1
    
j = 0
for i in tqdm(os.listdir(pituitary)):
    path = os.path.join(pituitary, i)
    img = cv2.imread(path)
    img = crop_image(img, plot=False)

    if img is not None:
        img = cv2.resize(img, (240, 240))
        save_path =  docelowe_dir + '/pituitary/' + str(j) + '.jpg'
        cv2.imwrite(save_path, img)
        j = j + 1