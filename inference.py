"""
For inference.
"""

import os 
import pickle

import skimage.io as io
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray
from skimage.transform import resize, rescale

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt

from patches_proc import backprojection
from utils import scsr, ScSR


dict_size = 1024
lmbd = 0.1              # lambda
patch_size = 3
overlap = 1             # degree of overlap between adjacent patches

num_samples = 100000
upscale = 2             # up-scaling factor
prune_perc = 10     # percentage for prunning
iter = 100
# lmbd = 0.3              # lambda 
# upsample = 2            # up-scaling 
color_space = 'ycbcr'   # color space  
max_iteration = 100     # maximum number of iterations for backprojection
nu = 1  
beta = 0 # c/gamma in paper

def infer(save=False): 

    dict_name = str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size)
    dh_path = 'data/dicts/Dh_' + dict_name + '.pkl'
    assert os.path.exists(dh_path), f"{dh_path} does not exist!"

    with open(dh_path, 'rb') as f:
        Dh = pickle.load(f)
    with open('data/dicts/Dl_' + dict_name + '.pkl', 'rb') as f:
        Dl = pickle.load(f)
 
    print('The dictionary we use:')
    print('data/dicts/Dh_' + dict_name + '.pkl')
    print('data/dicts/Dl_' + dict_name + '.pkl')

    img_lr_dir = 'data/val_lr/'

   

    img_lr_file = os.listdir(img_lr_dir)
    
    # INFERENCE
    for i in range(1): 
        img_name = img_lr_file[i]
        img_lr = io.imread(f"{img_lr_dir}/{img_name}")
        # img_hr = io.imread(f"{img_hr_dir}/{img_name}")

        if color_space == 'ycbcr':
            # img_hr_y = rgb2ycbcr(img_hr)[:,:,0]
            
            # Change color space from RGB to YCbCr
            
            img_lr_ycbcr = rgb2ycbcr(img_lr)
            img_lr_y = img_lr_ycbcr[:,:,0]

            img_lr_cb = img_lr_ycbcr[:,:,1]
            img_lr_cr = img_lr_ycbcr[:,:,2]
            
            # upscale chrominance to color SR images
            # nearest neighbor interpolation
            # img_sr_cb = resize(img_lr_cb, img_hr_y.shape, 0)
            # img_sr_cr = resize(img_lr_cr, img_hr_y.shape, 0)
            img_sr_cb = rescale(img_lr_cb, upscale, 0)
            img_sr_cr = rescale(img_lr_cr, upscale, 0)
            expected_shape = img_sr_cb.shape
        else:
            raise ValueError("Invalid color space!")

        img_sr_y = ScSR(img_lr_y, expected_shape, upscale, Dh, Dl, lmbd, overlap)
        img_sr_y = backprojection(img_sr_y, img_lr_y, max_iteration, nu, beta)


        if color_space == 'ycbcr':
            img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
            img_sr = ycbcr2rgb(img_sr)
            
        # elif color_space == 'bw':
        #     img_sr = img_sr_y
        else: 
            pass 
            ## value error will occur in previous phrase.

        io.imsave(f'results/infer_{img_name}_x{upscale}.png', img_sr)
        
        ## Bicubic Interpolation Image img_bc
        # img_bc = resize(img_lr, expected_shape, 3).clip(0,1)*255
        # img_bc = img_bc.astype(np.uint8)
        # img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]
        
        # calculate PSNR, SSIM and MSE for the luminance
        # img_hr_y = img_hr_y.astype(np.uint8)
        # img_bc_y = img_bc_y.astype(np.uint8)
        # img_sr_y = img_sr_y.astype(np.uint8)
        # for bicubic interpolation
        # psnr_bc_hr = peak_signal_noise_ratio(img_hr_y,img_bc_y)
        # ssim_bc_hr = structural_similarity(img_hr_y,img_bc_y)
        # mse_bc_hr = mean_squared_error(img_hr_y,img_bc_y)
        # nqm_bc_hr = noise_quality_measure(img_hr_y,img_bc_y)
        # for sparse representation SR
        # psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
        # ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
        # mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)