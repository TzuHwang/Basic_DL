import os, sys, natsort, json, random
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass

def apply_windowing(npy, w_center=400, w_width=1300, cvt2image=True, unnormalize_hu = False):
    if unnormalize_hu:
        npy = npy*4096-1024
    y_min = w_center - 0.5 * w_width
    y_max = w_center + 0.5 * w_width
    image = np.zeros_like(npy)
    image[npy < y_min] = 0.0
    image[npy > y_max] = 255.0

    in_between = np.logical_and(npy >= y_min, npy <= y_max)
    image[in_between] =\
        ((npy[in_between] - (w_center - 0.5)) /
         (w_width - 1.0) + 0.5) * (255.0 - 0.0) + 0.0
    if cvt2image is True:
        return Image.fromarray(np.array(image)).convert('RGBA') # CT_NP512S
    return np.array(image, dtype=np.int16)

def imgmerge(baseimg, coverimg, color = np.array([71, 130, 255, 60])):
    baseimg = Image.fromarray(baseimg)
    coverimg_draw = np.zeros(coverimg.shape+(4,)).astype('uint8')
    for i in range(4):
        coverimg_draw[:, :, i][coverimg > 0] = color[i]
    coverimg_draw = Image.fromarray(coverimg_draw)
    final = Image.alpha_composite(baseimg, baseimg)
    compose = Image.alpha_composite(final, coverimg_draw)
    return np.array(compose)

def mask_center(mask):
    ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    m = cv2.moments(thresh)
    x = int(m['m10']/m['m00'])
    y = int(m['m01']/m['m00'])
    return x, y

def mask_center_3D(masks):
    return center_of_mass(masks)

def fill_socket(image):
    image_ = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    isbreak = False
    for i in range(image_.shape[0]):
        for j in range(image_.shape[1]):
            if(image_[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    cv2.floodFill(image_, mask,seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(image_)
    image = image | im_floodfill_inv
    return image