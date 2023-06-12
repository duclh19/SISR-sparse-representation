"""
Implementation of a function sample_patches(), sample a number of patches from a single image. 
"""

import numpy as np
import skimage 
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from scipy.signal import convolve2d
from tqdm.auto import tqdm
from pprint import pprint

# def sample_patches(img, patch_size:int, patch_num, upscale):
    # """
    # Params: 
    #     img: input image to be sampled (high-resolution)
    #     patch_size: <int> size of a sampled patch in both directions (s, s)
    #     patch_num: <int> number of patches to be sampled
    #     upscale: <int> to generate low-resolution
    # ------
    # Return: 
    #     HP: <np.ndarray> high-resolution patches with shape (f_h, patch_num)
    #         f_h = patch_size * patch_size
    #     LP: <np.ndarray> los-resolution patches with shape (f_l, patch_num)
    #         f_l = 4 * patch_size * patch_size 
    # """
    # assert patch_num > 0, f"Number of patches should be higher than 0, here {patch_num}"
    # if img.shape[2] == 3:
    #     hIm = rgb2gray(img)
    # else:
    #     hIm = img

    # # Generate low resolution counter parts
    # lIm = rescale(hIm, 1 / upscale)
    # lIm = resize(lIm, hIm.shape)
    # nrow, ncol = hIm.shape

    # x = np.random.permutation(range(nrow - 2 * patch_size)) + patch_size
    # y = np.random.permutation(range(ncol - 2 * patch_size)) + patch_size

    # X, Y = np.meshgrid(x, y)
    # xrow = np.ravel(X, order='F')
    # ycol = np.ravel(Y, order='F')

    # if patch_num < len(xrow):
    #     xrow = xrow[0 : patch_num]
    #     ycol = ycol[0 : patch_num]

    # patch_num = len(xrow)

    # H = np.zeros((patch_size ** 2, len(xrow)))
    # L = np.zeros((4 * patch_size ** 2, len(xrow)))

    # # Compute the first and second order gradients
    # hf1 = [[-1, 0, 1], ] * 3
    # vf1 = np.transpose(hf1)

    # lImG11 = convolve2d(lIm, hf1, 'same')
    # lImG12 = convolve2d(lIm, vf1, 'same')

    # hf2 = [[1, 0, -2, 0, 1], ] * 3
    # vf2 = np.transpose(hf2)

    # lImG21 = convolve2d(lIm, hf2, 'same')
    # lImG22 = convolve2d(lIm, vf2, 'same')

    # for i in tqdm(range(patch_num)):
    #     row = xrow[i]
    #     col = ycol[i]

    #     Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size], order='F')
    #     # Hpatch = np.reshape(Hpatch, (Hpatch.shape[0], 1))
        
    #     Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size], order='F')
    #     Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
    #     Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size], order='F')
    #     Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
    #     Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size], order='F')
    #     Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
    #     Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size], order='F')
    #     Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))

    #     Lpatch = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4), axis=1)
    #     Lpatch = np.ravel(Lpatch, order='F')

    #     if i == 0:
    #         HP = np.zeros((Hpatch.shape[0], 1))
    #         LP = np.zeros((Lpatch.shape[0], 1))
    #         # print(HP.shape)
    #         HP[:, i] = Hpatch - np.mean(Hpatch)
    #         LP[:, i] = Lpatch
    #     else:
    #         HP_temp = Hpatch - np.mean(Hpatch)
    #         HP_temp = np.reshape(HP_temp, (HP_temp.shape[0], 1))
    #         HP = np.concatenate((HP, HP_temp), axis=1)
    #         LP_temp = Lpatch
    #         LP_temp = np.reshape(LP_temp, (LP_temp.shape[0], 1))
    #         LP = np.concatenate((LP, LP_temp), axis=1)
    
    # return HP, LP

def sample_patches(img, patch_size, patch_num, upscale_factor):
    if len(img.shape) == 3:
        hIm = (rgb2gray(img)*255).astype(np.uint8)
    else:
        hIm = img

    # Generate low resolution patches
    lIm = rescale(hIm, 1/upscale_factor, 3, preserve_range = True)
    lIm = resize(lIm, hIm.shape, 3, preserve_range = True)
    # lIm = lIm.astype(np.uint8)
    nrow, ncol = hIm.shape
    
    x = np.random.permutation(range(nrow - 2*patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2*patch_size)) + patch_size
    
    X,Y = np.meshgrid(x,y)
    xrow = np.ravel(X,'F')
    ycol = np.ravel(Y,'F')
    
    if patch_num<len(xrow):
        xrow = xrow[0:patch_num]
        ycol = ycol[0:patch_num]
        
    patch_num = len(xrow)
    HP = np.zeros([patch_size**2, patch_num])
    LP = np.zeros([4*patch_size**2, patch_num])
    
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')
    
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T
    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')
    
    for i in tqdm(range(patch_num)):
        row = xrow[i]
        col = ycol[i]
        
        Hpatch = np.ravel(hIm[row : row + patch_size, col : col + patch_size],'F')
        
        Lpatch1 = np.ravel(lImG11[row : row + patch_size, col : col + patch_size],'F')
        Lpatch1 = np.reshape(Lpatch1, (Lpatch1.shape[0], 1))
        Lpatch2 = np.ravel(lImG12[row : row + patch_size, col : col + patch_size],'F')
        Lpatch2 = np.reshape(Lpatch2, (Lpatch2.shape[0], 1))
        Lpatch3 = np.ravel(lImG21[row : row + patch_size, col : col + patch_size],'F')
        Lpatch3 = np.reshape(Lpatch3, (Lpatch3.shape[0], 1))
        Lpatch4 = np.ravel(lImG22[row : row + patch_size, col : col + patch_size],'F')
        Lpatch4 = np.reshape(Lpatch4, (Lpatch4.shape[0], 1))
        
        Lpatch = np.concatenate((Lpatch1,Lpatch2,Lpatch3,Lpatch4), axis = 1)
        Lpatch = np.ravel(Lpatch,'F')
        
        HP[:,i] = Hpatch - np.mean(Hpatch)
        LP[:,i] = Lpatch
        
    return HP, LP


if __name__ == "__main__":
    img = skimage.io.imread("data/test.JPG")
    patch_size = 4
    patch_num = 0
    upscale = 3
    HP, LP = sample_patches(img, patch_size, patch_num, upscale)

    print(HP.shape, LP.shape, type(HP))