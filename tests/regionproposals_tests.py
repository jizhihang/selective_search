from scipy.misc import imread, imsave
from scipy.ndimage.filters import gaussian_filter

from regionproposals.regionproposals import segment, rgb2hsv, nms

import numpy

import time

def rgb2hsv_test():
    image = imread('tests/08541b84c3f1b1600d094507010e1882.jpg')
    hsv = rgb2hsv(image)

def regionproposal_test():
    image = imread('tests/08541b84c3f1b1600d094507010e1882.jpg')

    sigma = 0.5

    tic = time.time()
    image = gaussian_filter(image, sigma)
    toc = time.time()
    print(toc - tic)

    image = rgb2hsv(image)
    imsave('hsv.png', image.astype(numpy.uint8))

    tic = time.time()
    segmentation, regions = segment(image, c=300, min_size=20, range1=255, range2=255, range3=255)
    toc = time.time()
    print(toc - tic)

    print(len(regions))

    for left, right, top, bottom, label in regions:
        segmentation[top:bottom+1,left,:] = (255, 0, 0)
        segmentation[top:bottom+1,right,:] = (255, 0, 0)
        segmentation[top,left:right+1,:] = (255, 0, 0)
        segmentation[bottom,left:right+1,:] = (255, 0, 0)
    imsave('segmentation.png', segmentation)
