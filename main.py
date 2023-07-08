import os 
import pickle

import numpy as np 
import skimage.io as io

from matplotlib import pyplot as plt
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.transform import resize
from sklearn.preprocessing import normalize

from config import EvalConfig
from utils import ScSR
from patches_proc import backprojection


# PARAMETERS SET UP
params = EvalConfig()


with open(os.path.join(params.dictionary_path, params.hr_dict), 'rb') as f:
    Dh = pickle.load(f)
# Dh = normalize(Dh)
with open(os.path.join(params.dictionary_path, params.lr_dict), 'rb') as f:
    Dl = pickle.load(f)
# Dl = normalize(Dl)

print('The dictionary we use:')
print(f'\tHR dictionary: {params.hr_dict}')
print(f'\tLR dictionary: {params.lr_dict}')


img_lr_file = os.listdir(params.lr_eval_path)

# for i in range(len(img_lr_file[1])): 
for i in range(1): 
    img_name = img_lr_file[i]
    img_lr = io.imread(os.path.join(params.lr_eval_path, img_lr_file[i]))
    img_hr = io.imread(os.path.join(params.hr_eval_path, img_lr_file[i]))

    if params.color_space == 'ycbcr':
        img_hr_y = rgb2ycbcr(img_hr)[:,:,0]
        
        # Change color space
        img_lr_ycbcr = rgb2ycbcr(img_lr)
        img_lr_y    = img_lr_ycbcr[:,:,0]
        img_lr_cb   = img_lr_ycbcr[:,:,1]
        img_lr_cr   = img_lr_ycbcr[:,:,2]
        
        # Upscale chrominance to color SR images
        # nearest neighbor interpolation

        img_sr_cb = resize(img_lr_cb, img_hr_y.shape, 0)
        img_sr_cr = resize(img_lr_cr, img_hr_y.shape, 0)
        
    else:
        raise ValueError("Invalid color space!")

    # Step 2 and 3 in Algorithm 1 from paper
    img_sr_y = ScSR(img_lr_y, img_hr_y.shape, params.upscale, Dh, Dl, params.lmbda, params.overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, params.max_iter, params.nu, params.beta)

    if params.color_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)

        
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
    # for sparse representation SR
    psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
    ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
    mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)
    

    ### NOW, img_bc, img_hr, img_sr should be saved in one figure. 
    fig, ax = plt.subplots(1, 3, sharey=False)
    # plt.subplot(1, 2, 1, fig)
    ax[0].imshow(img_bc)
    ax[0].set_title('bicubic')
    ax[1].imshow(img_hr)
    ax[1].set_title('original high resolution')

    ax[2].imshow(img_sr)
    ax[2].set_title('super-resolution from low-res')
    plt.savefig(f'{img_name}_result.png')
    print(img_name)
    print(f'psnr -> bc: {psnr_bc_hr}, sr: {psnr_sr_hr}')
    print(f'ssim -> bc: {ssim_bc_hr}, sr: {ssim_sr_hr}')
    print(f'mse -> bc: {mse_bc_hr}, sr: {mse_sr_hr}')



