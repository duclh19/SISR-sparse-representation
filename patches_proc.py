"""
Implementation of some functions processing image patches.
    
"""
import os
import numpy as np 
from tqdm.auto import tqdm
import skimage.io as io
from skimage.transform import resize, rescale
from skimage.color import rgb2gray

from scipy.signal import convolve2d

def sample_patches(img, patch_size, patch_num, upscale):
    print(img.shape)
    if img.shape[2] == 3:
        hIm = rgb2gray(img)
    else:
        hIm = img

    # Generate low resolution counter parts
    lIm = rescale(hIm, 1 / upscale)
    lIm = resize(lIm, hIm.shape)
    nrow, ncol = hIm.shape
    print('rnow, ncol', nrow, ncol)

    x = np.random.permutation(range(nrow - 2 * patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2 * patch_size)) + patch_size
    print(x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    xrow = np.ravel(X, order='F')
    ycol = np.ravel(Y, order='F')

    if patch_num < len(xrow):
        xrow = xrow[0 : patch_num]
        ycol = ycol[0 : patch_num]

    patch_num = len(xrow)

    H = np.zeros((patch_size ** 2, len(xrow)))
    L = np.zeros((4 * patch_size ** 2, len(xrow)))

    # Compute the first and second order gradients
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')

    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')

    for i in tqdm(range(patch_num)):
        row = xrow[i]
        col = ycol[i]
        # print(hIm[row : row + patch_size, col : col + patch_size].shape)
        Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size], order='F')
        
        Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size], order='F')
        
        Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
        # print(Lpatch1.shape)
        # return
        Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
        Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
        Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size], order='F')
        Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))

        Lpatch = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4), axis=1)
        Lpatch = np.ravel(Lpatch, order='F')

        if i == 0:
            HP = np.zeros((Hpatch.shape[0], 1))
            LP = np.zeros((Lpatch.shape[0], 1))
            # print(HP.shape)
            HP[:, i] = Hpatch - np.mean(Hpatch)
            LP[:, i] = Lpatch
        else:
            HP_temp = Hpatch - np.mean(Hpatch)
            HP_temp = np.reshape(HP_temp, (HP_temp.shape[0], 1))
            HP = np.concatenate((HP, HP_temp), axis=1)
            LP_temp = Lpatch
            LP_temp = np.reshape(LP_temp, (LP_temp.shape[0], 1))
            LP = np.concatenate((LP, LP_temp), axis=1)
    
    return HP, LP

def random_sample_patch(img_path, patch_size, num_patch, upsample):
    
    img_dir = os.listdir(img_path)
    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in (range(img_num)):
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]

    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img, axis=0))

    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = io.imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upsample)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)
    return Xh, Xl

def patch_pruning(Xh, Xl, per=10):
    """
    Parameters: 
    -----
    `Xh`: \\
    `Xl`: \\
    `per`: pruning percentage 
        if 100 -> prun all 
        if 0 -> no prunning

    Return:
    ------
    `Xh`\\
    `Xl`
    """
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, per)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl

def gauss2D(shape:int, sigma:float):
    """
    Gaussian filter with `shape` and `sigma`

    Parameters
    ------
    `shape`: \\ 
    `sigma`:

    Return: 
    -----
    `h`: 2-D array which is a Gaussian filter 
    """
    m,n = [(ss-1.)/2. for ss in (shape, shape)]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def backprojection(sr:np.ndarray, lr:np.ndarray, iters:int, nu:float, c:float):
    """
    Back projection: step 3 in Algorithm 1 from the paper

    Parameters
    -----
    `sr`:
    `lr`:\\
    `iters`: maximum number of iterations (until convergence -> 10e5 maybe)
    `nu`: ...
    `c`: c in Algorithm 1. 

    Return: 
    -----
    `sr`: optimal super-resolution image
    """
    p = gauss2D(5,1)
    p = p*p
    p = p/np.sum(p)
    
    sr = sr.astype(np.float64)
    lr = lr.astype(np.float64)
    sr_0 = sr
    
    for i in range(iters):
        #sr_blur = convolve2d(sr, p, 'same')
        sr_downscale = resize(sr, lr.shape)
        diff = lr - sr_downscale

        diff_upscale = resize(diff, sr_0.shape)
        diff_blur = convolve2d(diff_upscale, p, 'same')
        
        sr = sr + nu*(diff_blur + c*(sr_0-sr))
    return sr

if __name__ == "__main__": 
    pass
