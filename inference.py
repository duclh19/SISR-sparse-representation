"""
For inference.
"""

import os 
import pickle

import skimage.io as io
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.transform import resize, rescale


import numpy as np 
from matplotlib import pyplot as plt
from config import TestConfig

from patches_proc import backprojection
from utils import ScSR


def infer(params:TestConfig, save=False): 
    assert os.path.isfile(params.image_path), f"{params.image_path} does not exist!"
    assert os.path.exists(params.lr_dict), f"{params.lr_dict} does not exist!"
    assert os.path.exists(params.hr_dict), f"{params.hr_dict} does not exist!"

    with open(params.hr_dict, 'rb') as f:
        Dh = pickle.load(f)
    with open(params.lr_dict, 'rb') as f:
        Dl = pickle.load(f)
 
    print('The dictionary we use:')
    print(params.lr_dict)
    print(params.hr_dict)

    # INFERENCE
    # for i in range(1): 
    img_lr = io.imread(f'{params.image_path}')

    # Change color space from RGB to YCbCr
    img_lr_ycbcr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr_ycbcr[:,:,0]

    img_lr_cb = img_lr_ycbcr[:,:,1]
    img_lr_cr = img_lr_ycbcr[:,:,2]
    
    img_sr_cb = rescale(img_lr_cb, params.upscale, 0)
    img_sr_cr = rescale(img_lr_cr, params.upscale, 0)
    expected_shape = img_sr_cb.shape

    img_sr_y = ScSR(img_lr_y, expected_shape, params.upscale, Dh, Dl, params.lmbda, params.overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, 1000, params.nu, params.beta)


    img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr = ycbcr2rgb(img_sr)
        
    name = f'x{params.upscale}_{os.path.basename(params.image_path)}'
    return img_sr, name
        
if __name__ == "__main__": 
    if not os.path.exists("result/"): 
        os.mkdir('result/')
        print('Create result/')

    params = TestConfig(image_path='face.png')
    img_sr, name = infer(params=params)
    io.imsave(os.path.join('result', name), img_sr)