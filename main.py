"""
Validation? 
"""
import os 
import pickle

import skimage.io as io
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray
from skimage.transform import resize

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt
# from matplotlib import style
# from matplotlib import rcParams
from patches_proc import backprojection
from utils import scsr, ScSR

# PARAMETERS SET UP

dict_size = 1024
lmbd = 0.1
patch_size = 3
num_samples = 100000
upscale = 2
prune_perc = 10 # percentage for prunning
iter = 100
# train_img_path = './data/T91/'

dict_name = str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size)

with open('data/dicts/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
# Dh = normalize(Dh)
with open('data/dicts/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
# Dl = normalize(Dl)
print('The dictionary we use:')
print('data/dicts/Dh_' + dict_name + '.pkl')
print('data/dicts/Dl_' + dict_name + '.pkl')

img_lr_dir = 'data/val_lr/'
img_hr_dir = 'data/val_hr/'

overlap = 1
lmbd = 0.3
upsample = 2
color_space = 'ycbcr'
# color_space = 'bw'
max_iteration = 100
nu = 1
beta = 0 

img_lr_file = os.listdir(img_lr_dir)

# for i in range(len(img_lr_file[1])): 
for i in range(1): 
    img_name = img_lr_file[i]
    img_lr = io.imread(f"{img_lr_dir}/{img_name}")
    img_hr = io.imread(f"{img_hr_dir}/{img_name}")

    if color_space == 'ycbcr':
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
        
    elif color_space == 'bw':
        img_hr_y = rgb2gray(img_hr)
        img_lr_y = rgb2gray(img_lr)
    
    else:
        raise ValueError("Invalid color space!")

    img_sr_y = ScSR(img_lr_y, img_hr_y.shape, upsample, Dh, Dl, lmbd, overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, max_iteration, nu, beta)

    if color_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)
        
    elif color_space == 'bw':
        img_sr = img_sr_y
        
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
    # nqm_bc_hr = noise_quality_measure(img_hr_y,img_bc_y)
    # for sparse representation SR
    psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
    ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
    mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)
    # nqm_sr_hr = noise_quality_measure(img_hr_y,img_sr_y)
    
    print(img_name)
    print(f'psnr -> bc: {psnr_bc_hr}, sr: {psnr_sr_hr}')
    print(f'ssim -> bc: {ssim_bc_hr}, sr: {ssim_sr_hr}')
    print(f'mse -> bc: {mse_bc_hr}, sr: {mse_sr_hr}')
    # print(f'nqm -> bc: {nqm_bc_hr}, sr: {nqm_sr_hr}')

### OLD 
# for i in tqdm(range(len(img_lr_file))):
# # for i in tqdm(range(1)):
#     # READ IMAGE AND MAKE FOLDERS
#     img_name = img_lr_file[i]
#     img_name_dir = list(img_name)
#     img_name_dir = np.delete(np.delete(np.delete(np.delete(img_name_dir, -1), -1), -1), -1)
#     img_name_dir = ''.join(img_name_dir)
#     if os.path.isdir('data/results/set5_sigma25/' + dict_name + '_' + img_name_dir) == False:
#         new_dir = os.mkdir('{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir))
#     img_lr = io.imread('{}{}'.format(img_lr_dir, img_name))

#     ## READ AND SAVE ORIGINAL IMAGE
#     img_hr = io.imread('{}{}'.format(img_hr_dir, img_name))
#     io.imsave('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', '3HR.png'), img_hr)
    
#     if color_space == 'ycbcr':
#         img_hr_y = rgb2ycbcr(img_hr)[:,:,0]

#         ## CHANGE COLOUR SPACE
#         img_lr_ori = img_lr
#         temp = img_lr
#         imr_lr = rgb2ycbcr(img_lr)
#         img_lr_y = img_lr[:,:,0]
#         img_lr_cb = img_lr[:,:,1]
#         img_lr_cr = img_lr[:,:,2]

#         ## UPSAMPLE CHROMINANCE CHANNEL DIRECTLY
#         img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
#         img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)
    
#     elif color_space == 'bw':
#         img_hr_y = rgb2gray(img_hr)
#         img_lr = rgb2gray(img_lr)
#         img_lr_y = img_lr
#         img_lr_ori = img_lr

#     ## SUPER-RESOLUTION OF LUMINANCE CHANNEL
#     img_sr_y = ScSR(img_lr_y, img_hr_y.shape, upsample, Dh, Dl, lmbd, overlap)
#     img_sr_y = backprojection(img_sr_y, img_lr_y, max_iteration)
#     # img_sr_y = resize(img_lr_y, (img_hr.shape[0], img_hr.shape[1]), order=0) # Loop check

#     ## RECONSTRUCT COLOUR IMAGE
#     if color_space == 'ycbcr':
#         img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
#         img_sr = ycbcr2rgb(img_sr)
        
#         for channel in range(img_sr.shape[2]):
#             img_sr[:, :, channel] = normalize_signal(img_sr, img_lr_ori, channel)

#         img_sr = normalize_max(img_sr)
    
#     elif color_space == 'bw':
#         img_sr = img_sr_y

#     ## COMPUTE METRICS
#     rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
#     # psnr_sr_hr = 10*np.log10(255.0**2/rmse_sr_hr**2)
#     psnr_sr_hr = 10*np.log10(1.0**2/rmse_sr_hr**2)
#     psnr_sr_hr = np.zeros((1,)) + psnr_sr_hr
#     np.savetxt('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', 'PSNR_SR.txt'), psnr_sr_hr)

#     ## SAVE SUPER-RESOLVED IMAGE
#     io.imsave('{}{}{}{}'.format('data/results/set5_sigma25/' + dict_name + '_', img_name_dir, '/', '2SR.png'), img_sr)

# %% PLOTS

# fig, plts = plt.subplots(1,2,figsize=(10,6))
# plts[0].imshow(img_sr_y, cmap='gray')
# plts[0].set_title(r"Super-Resolved Lena PSNR: %.4f"%(psnr_sr_hr))

# plts[1].imshow(img_hr_y, cmap='gray')
# plts[1].set_title(r"Original Lena PSNR")

# # plt.show()



# fig, axs = plt.subplots(4,5, figsize=(15, 12), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace = .5, wspace=.001)
# axs = axs.ravel()
# for i in range(20):
#     axs[i].imshow(Dl[:,i].reshape(6,6),cmap='gray')
#     axs[i].axis('off')

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Course Project/Slides/figures/dictionary.eps', format='eps')
# plt.show()


import os 
import pickle

import skimage.io as io
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray
from skimage.transform import resize

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt
# from matplotlib import style
# from matplotlib import rcParams
from patches_proc import backprojection
from utils import scsr, ScSR

# PARAMETERS SET UP

dict_size = 1024
lmbd = 0.1
patch_size = 3
num_samples = 100000
upscale = 2
prune_perc = 10 # percentage for prunning
iter = 100
# train_img_path = './data/T91/'

dict_name = str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size)

with open('data/dicts/Dh_' + dict_name + '.pkl', 'rb') as f:
    Dh = pickle.load(f)
# Dh = normalize(Dh)
with open('data/dicts/Dl_' + dict_name + '.pkl', 'rb') as f:
    Dl = pickle.load(f)
# Dl = normalize(Dl)
print('The dictionary we use:')
print('data/dicts/Dh_' + dict_name + '.pkl')
print('data/dicts/Dl_' + dict_name + '.pkl')

img_lr_dir = 'data/val_lr/'
img_hr_dir = 'data/val_hr/'

overlap = 1
lmbd = 0.3
upsample = 2
color_space = 'ycbcr'
# color_space = 'bw'
max_iteration = 100
nu = 1
beta = 0 

img_lr_file = os.listdir(img_lr_dir)

# for i in range(len(img_lr_file[1])): 
for i in range(1): 
    img_name = img_lr_file[i]
    img_lr = io.imread(f"{img_lr_dir}/{img_name}")
    img_hr = io.imread(f"{img_hr_dir}/{img_name}")

    if color_space == 'ycbcr':
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
        
    elif color_space == 'bw':
        img_hr_y = rgb2gray(img_hr)
        img_lr_y = rgb2gray(img_lr)
    
    else:
        raise ValueError("Invalid color space!")

    img_sr_y = ScSR(img_lr_y, img_hr_y.shape, upsample, Dh, Dl, lmbd, overlap)
    img_sr_y = backprojection(img_sr_y, img_lr_y, max_iteration, nu, beta)

    if color_space == 'ycbcr':
        img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
        img_sr = ycbcr2rgb(img_sr)
        
    elif color_space == 'bw':
        img_sr = img_sr_y
        
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
    # nqm_bc_hr = noise_quality_measure(img_hr_y,img_bc_y)
    # for sparse representation SR
    psnr_sr_hr = peak_signal_noise_ratio(img_hr_y,img_sr_y)
    ssim_sr_hr = structural_similarity(img_hr_y,img_sr_y)
    mse_sr_hr = mean_squared_error(img_hr_y,img_sr_y)
    # nqm_sr_hr = noise_quality_measure(img_hr_y,img_sr_y)
    

    ### NOW, img_bc, img_hr, img_sr should be saved in one figure. 
    fig, ax = plt.subplots(1, 3, sharey=False)
    # plt.subplot(1, 2, 1, fig)
    ax[0].imshow(img_bc)
    ax[1].imshow(img_hr)
    ax[2].imshow(img_sr)
    plt.savefig(f'{img_name}_result.png')
    print(img_name)
    print(f'psnr -> bc: {psnr_bc_hr}, sr: {psnr_sr_hr}')
    print(f'ssim -> bc: {ssim_bc_hr}, sr: {ssim_sr_hr}')
    print(f'mse -> bc: {mse_bc_hr}, sr: {mse_sr_hr}')
    # print(f'nqm -> bc: {nqm_bc_hr}, sr: {nqm_sr_hr}')