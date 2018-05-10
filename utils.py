#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 07:36:23 2018

@author: chocolatethunder
"""
import numpy as np
import cv2
from mxnet import ndarray as nd
from mxnet import autograd

def save_image(ims, filename):
    
    height = ims[0].shape[1]
    
    padding = np.zeros((3,height,5))
    
    im = []
    
    for x in ims[:-1 ]:
        im.append(x)
        im.append(padding)
    im.append(ims[-1])
    im = np.concatenate(tuple(im), axis = 2).swapaxes(0,2).swapaxes(0,1)
    im *=255
    cv2.imwrite(filename, im)
    
def binarize(pred):
    
    length = pred.shape[0]
    pred = pred.reshape((length,1))
    binary_pred = nd.zeros((64,2))
    for ii,x in enumerate(pred) :
        binary_pred[ii,0] = 1-x
        binary_pred[ii,1] = x
    return binary_pred