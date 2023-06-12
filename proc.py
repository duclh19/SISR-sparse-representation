"""
This code is used to process images 

"""

import numpy as np 
from skimage.transform import rescale
from skimage.io import imread, imsave
import os
from os import listdir
from tqdm import tqdm

# Set train and val HR and LR paths
train_hr_path = 'data/T91/'
train_lr_path = 'data/T91_lr/'
val_hr_path = 'data/val_hr/'
val_lr_path = 'data/val_lr/'


if __name__ == "__main__": 
    if not os.path.exists(train_lr_path): 
        os.makedirs(train_lr_path)
    if not os.path.exists(val_lr_path): 
        os.makedirs(val_lr_path)
    numberOfImagesTrainHR = len(listdir(train_hr_path))

    for i in tqdm(range(numberOfImagesTrainHR)):
        img_name = listdir(train_hr_path)[i]
        img = imread('{}{}'.format(train_hr_path, img_name))
        new_img = rescale(img, (1/3), anti_aliasing=1, channel_axis=-1)
        imsave('{}{}'.format(train_lr_path, img_name), (new_img*255).astype(np.uint8))

    numberOfImagesValHR = len(listdir(val_hr_path))

    for i in tqdm(range(numberOfImagesValHR)):
        img_name = listdir(val_hr_path)[i]
        img = imread('{}{}'.format(val_hr_path, img_name))
        new_img = rescale(img, (1/3), anti_aliasing=1, channel_axis=-1)
        imsave('{}{}'.format(val_lr_path, img_name), (new_img*255).astype(np.uint8))