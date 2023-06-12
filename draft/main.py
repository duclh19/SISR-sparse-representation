import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2ycbcr,ycbcr2rgb,rgb2gray
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from utils import noise_quality_measure

from utils import scsr, backprojection
import pickle

# choose parameters
class args(object):
    lr_dir = 'data/set5_lr'
    sr_dir = 'result/set5_sr'
    hr_dir = 'data/set5_hr'
    
    # choose a dictionary for SR
    dic_upscale_factor = 2
    dic_lambda = 0.1
    dic_size = 1024
    dic_patch_size = 3
    
    # sparse SR factor
    lambda_factor = 0.1
    overlap = 2
    upscale_factor = 2
    max_iteration = 100
    nu = 1
    beta = 0 # also c/gamma in paper
    color_space = 'ycbcr' # 'bw'
    
    # True for validation, False for prediction
    val_flag = True
    
para = args()



# Load dictionaries
dict_name = str(para.dic_size) + '_US' + str(para.dic_upscale_factor) + '_L' + str(para.dic_lambda) + '_PS' + str(para.dic_patch_size)

with open('dictionary/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
with open('dictionary/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
# import scipy.io as scio
 
# dataFile = './dictionary/D_512_0.15_5.mat'
# data = scio.loadmat(dataFile)
# Dh = data['Dh']
# Dl = data['Dl']

# %%
# super resolution img dir
if not os.path.exists(para.sr_dir):
    os.makedirs(para.sr_dir)

img_lr_file = os.listdir(para.lr_dir)



for i in range(len(img_lr_file)):
    img_name = img_lr_file[i]
    img_lr = io.imread(f"{para.lr_dir}/{img_name}")
    img_hr = io.imread(f"{para.hr_dir}/{img_name}")
    
    if para.color_space == 'ycbcr':
        img_hr_y = rgb2ycbcr(img_hr)[:,:,0]
        
        # change color space
        img_lr_ycbcr = rgb2ycbcr(img_lr)
        img_lr_y = img_lr_ycbcr[:,:,0]
        img_lr_cb = img_lr_ycbcr[:,:,1]
        img_lr_cr = img_lr_ycbcr[:,:,2]
        
        # upscale chrominance to color SR images
        # nearest neighbor interpolation
        img_sr_cb = resize(img_lr_cb, img_hr_y.shape, 0)
        img_sr_cr = resize(img_lr_cr, img_hr_y.shape, 0)
        
    elif para.color_space == 'bw':
        img_hr_y = rgb2gray(img_hr)
        img_lr_y = rgb2gray(img_lr)
    
    else:
        raise ValueError("Invalid color space!")
        
    # super resolution via sparse representation
    # TODO ScSR, backprojection
    #img_sr_y = scsr(img_lr_y, para.upscale_factor, Dh, Dl, para.lambda_factor, para.overlap, para.max_iteration)
    img_sr_y = resize(img_lr_y, np.multiply(para.upscale_factor, img_lr_y.shape))
    
    img_sr_y = backprojection(img_sr_y, img_lr_y, para.max_iteration, para.nu, para.beta)
    
    # reconstructed color images
    if para.color_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)
        
    elif para.color_space == 'bw':
        img_sr = img_sr_y
        
    else:
        raise ValueError("Invalid color space!")

    # # pixel intensity normalization
    img_sr = img_sr.clip(0,1)*255
    img_sr = img_sr.astype(np.uint8)
    
    # bicubic interpolation for reference
    img_bc = resize(img_lr, img_hr_y.shape, 3).clip(0,1)*255
    img_bc = img_bc.astype(np.uint8)
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]
    
    # calculate PSNR, SSIM and MSE for the luminance
    img_hr_y = img_hr_y.astype(np.uint8)
    img_bc_y = img_bc_y.astype(np.uint8)
    img_sr_y = img_sr_y.astype(np.uint8)
    # for bicubic interpolation
    psnr_bc_hr = peak_signal_noise_ratio(img_hr_y,img_bc_y)
    ssim_bc_hr = structural_similarity(img_hr_y,img_bc_y)
    mse_bc_hr = mean_squared_error(img_hr_y,img_bc_y)
    nqm_bc_hr = noise_quality_measure(img_hr_y,img_bc_y)
    # for sparse representation SR
    psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
    ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
    mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)
    nqm_sr_hr = noise_quality_measure(img_hr_y,img_sr_y)
    
    print(img_name)
    print(f'psnr -> bc: {psnr_bc_hr}, sr: {psnr_sr_hr}')
    print(f'ssim -> bc: {ssim_bc_hr}, sr: {ssim_sr_hr}')
    print(f'mse -> bc: {mse_bc_hr}, sr: {mse_sr_hr}')
    print(f'nqm -> bc: {nqm_bc_hr}, sr: {nqm_sr_hr}')
    
    # TODO :
    # Save images
    # Save scores

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(img_bc)
    # plt.subplot(1,2,2)
    # plt.imshow(img_sr)
    # plt.show()