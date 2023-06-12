"""
Implementation of a random sample patches named rnd_smp_patch()
    that randomly sample a number of patches from a image which belons a set of images. 
"""

from pprint import pprint
import numpy as np 
from os import listdir
from skimage.io import imread
from sample_patches import sample_patches
from tqdm import tqdm
def random_sample_patch(img_path, patch_size, num_patch, upscale_factor):
    img_dir = listdir(img_path)

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]
    
    nper_img = np.floor(nper_img * num_patch/np.sum(nper_img))
    
    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        img = imread('{}{}'.format(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upscale_factor)
        if i == 0:
            Xh = H
            Xl = L
        else:
            Xh = np.concatenate((Xh, H), axis=1)
            Xl = np.concatenate((Xl, L), axis=1)
    return Xh, Xl

def rnd_smp_patch(img_path, patch_size, num_patch, upscale):
    img_dir = listdir(img_path)
    print(f"number of images: {len(img_dir)}")
    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    for i in tqdm(range(img_num)):
        img = imread('{}{}'.format(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]
        print(img.shape[0], img.shape[1])
    print("check nper_img", nper_img, nper_img.shape)
    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img, axis=0))
    print("check nper_img", nper_img, nper_img.shape)

    ### Due to shape of original image
    ### some images might not be extracted patches
    drop = 0    # count number of images are not extracted

    for i in tqdm(range(img_num)):
        patch_num = int(nper_img[i])
        if patch_num > 0:
            img = imread('{}{}'.format(img_path, img_dir[i]))
            
            H, L = sample_patches(img, patch_size, patch_num, upscale)
            # except: 
                # print(img_dir[i])
            if i == 0:
                Xh = H
                Xl = L
            else:
                Xh = np.concatenate((Xh, H), axis=1)
                Xl = np.concatenate((Xl, L), axis=1)
        else: 
            drop +=1
            # print(Xh.shape)
    # patch_path = 
    print(f"Number of images that do not extracted {drop}")
    return Xh, Xl



def patch_pruning(Xh, Xl, per):
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, per)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl

if __name__ == "__main__": 
    img_path = 'data/'
    patch_size = 3
    num_patch = 12
    upscale = 3
    Xh, Xl = rnd_smp_patch(img_path, patch_size, num_patch, upscale)
    print(Xh.shape, Xl.shape, type(Xl))

    Xh, Xl = (patch_pruning(Xh, Xl))

    pprint(Xh.shape)
