#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:10:57 2018

@author: chocolatethunder
"""

import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from math import floor

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pix2pix_split_images(root,x):
    #x is either train, val or test
    paths = glob(os.path.join(root, x+"/*"))

    a_path = os.path.join(root, x+"_A/im")
    b_path = os.path.join(root, x+"_B/im")
    

    makedirs(a_path)
    makedirs(b_path)

    for path in tqdm(paths, desc="pix2pix processing"):
        filename = os.path.basename(path)

        a_image_path = os.path.join(a_path, filename)
        b_image_path = os.path.join(b_path, filename)

        if os.path.exists(a_image_path) and os.path.exists(b_image_path):
            continue

        image = Image.open(os.path.join(path)).convert('RGB')
        data = np.array(image)

        height, width, channel = data.shape

        a_image = Image.fromarray(data[:,:floor(width/2)].astype(np.uint8))
        b_image = Image.fromarray(data[:,floor(width/2):].astype(np.uint8))

        a_image.save(a_image_path)
        b_image.save(b_image_path)


if __name__ == "__main__":
    
    
    PIX2PIX_DATASETS = [
    'facades', 'cityscapes', 'maps','edges2shoes', 'edges2handbags']
    
    for dataset in PIX2PIX_DATASETS :
        root =  'Data/' + dataset
        pix2pix_split_images(root,'train')
        pix2pix_split_images(root,'val')
    