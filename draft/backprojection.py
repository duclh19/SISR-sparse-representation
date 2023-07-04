"""
Implementation of backprojection function, final step of Algorithm 1 by Yang et al. 
"""


import numpy as np 
from skimage.transform import resize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from pprint import pprint

def gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    -------
    return: 2D gaussian mask with shape=shape
    """
    m,n = [(ss-1.)/2. for ss in (shape, shape)]
    print(m, n)
    ### y = [[-m], [-m+1], .. .[m-1], [m]]
    ### x = [[-n, -n+1, ..., n-1, n]]
    y,x = np.ogrid[-m:m+1,-n:n+1]


    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def backprojection(sr, lr, iters, nu, c):
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

# def backprojection(img_hr, img_lr, maxIter):
#     """
#     Function for computing X* in step 3 in Algorithm 1. 
    
#     Params: 
#         img_hr: represent X0 after training step 2 in Algorithm 1. 
#         img_lr: represent Y 
#         maxIter: hyper-parameter --> until converges
    
#     -------
#     Return:
#         optimal high-resolution image X*
#     """
#     p = gauss2D((5, 5), 1)
#     p = np.multiply(p, p)       ## element-wise
#     p = np.divide(p, np.sum(p)) ## rescale

#     for i in range(maxIter):
#         img_lr_ds = resize(img_hr, img_lr.shape, anti_aliasing=1)
#         img_diff = img_lr - img_lr_ds

#         img_diff = resize(img_diff, img_hr.shape)
#         img_hr += convolve2d(img_diff, p, 'same')
#     return img_hr


# if __name__ == "__main__": 
#     shape = (5, 3)
#     sigma_1 = 1
#     pprint(gauss2D(shape, sigma_1).shape)

#     p = gauss2D((5, 5), 1)
#     pprint(p)
#     p = np.multiply(p, p)
#     pprint(p)
#     p = np.divide(p, np.sum(p))
    # pprint(p)

