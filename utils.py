import numpy as np 
from os import listdir
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage.color import rgb2ycbcr
from skimage.transform import resize
import pickle
from featuresign import fss_yang
from scipy.signal import convolve2d
from tqdm import tqdm

def noise_quality_measure(hr, sr, VA=np.pi/3):
    # metrics: noise quality measure 

    def ctf(f_r):
        y = 1/(200*2.6*(0.0192+0.114*f_r)*np.exp(-(0.114*f_r)**1.1))
        return y

    def cmaskn(c, ci, a, ai, i):
        cx = deepcopy(c)
        cix = deepcopy(ci)
        cix[np.abs(cix) > 1] = 1
        ct = ctf(i)
        T = ct*(.86*((cx/ct)-1)+.3)
        ai[(abs(cix-cx)-T) < 0] = a[(abs(cix-cx)-T) < 0]
        return ai

    def gthresh(x, T, z):
        z[np.abs(x) < T] = 0
        return z

    row, col = sr.shape
    X = np.linspace(-row/2+0.5, row/2-0.5, row)
    Y = np.linspace(-col/2+0.5, col/2-0.5, col)
    x, y = np.meshgrid(X, Y)
    plane = (x+1j*y)
    r = np.abs(plane)

    pi = np.pi
    
    G_0 = 0.5*(1+np.cos(pi*np.log2((r+2)*(r+2 >= 1) *(r+2 <= 4)+4*(~((r+2 <= 4)*(r+2 >= 1))))-pi))

    G_1 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 1)*(r <= 4))+4*(~((r >= 1)*(r <= 4))))-pi))

    G_2 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 2)*(r <= 8))+.5*(~((r >= 2) * (r <= 8))))))

    G_3 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 4)*(r <= 16))+4*(~((r >= 4)*(r <= 16))))-pi))

    G_4 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 8)*(r <= 32))+.5*(~((r >= 8) * (r <= 32))))))

    G_5 = 0.5*(1+np.cos(pi*np.log2(r*((r >= 16)*(r <= 64))+4*(~((r >= 16)*(r <= 64))))-pi))
    
    GS_0 = fftshift(G_0)
    GS_1 = fftshift(G_1)
    GS_2 = fftshift(G_2)
    GS_3 = fftshift(G_3)
    GS_4 = fftshift(G_4)
    GS_5 = fftshift(G_5)

    FO = fft2(sr).T
    FI = fft2(hr).T

    L_0 = GS_0*FO
    LI_0 = GS_0*FI

    l_0 = np.real(ifft2(L_0))
    li_0 = np.real(ifft2(LI_0))

    A_1 = GS_1*FO
    AI_1 = GS_1*FI

    a_1 = np.real(ifft2(A_1))
    ai_1 = np.real(ifft2(AI_1))

    A_2 = GS_2*FO
    AI_2 = GS_2*FI

    a_2 = np.real(ifft2(A_2))
    ai_2 = np.real(ifft2(AI_2))

    A_3 = GS_3*FO
    AI_3 = GS_3*FI

    a_3 = np.real(ifft2(A_3))
    ai_3 = np.real(ifft2(AI_3))

    A_4 = GS_4*FO
    AI_4 = GS_4*FI

    a_4 = np.real(ifft2(A_4))
    ai_4 = np.real(ifft2(AI_4))

    A_5 = GS_5*FO
    AI_5 = GS_5*FI

    a_5 = np.real(ifft2(A_5))
    ai_5 = np.real(ifft2(AI_5))

    c1 = a_1/l_0
    c2 = a_2/(l_0+a_1)
    c3 = a_3/(l_0+a_1+a_2)
    c4 = a_4/(l_0+a_1+a_2+a_3)
    c5 = a_5/(l_0+a_1+a_2+a_3+a_4)

    ci1 = ai_1/li_0
    ci2 = ai_2/(li_0+ai_1)
    ci3 = ai_3/(li_0+ai_1+ai_2)
    ci4 = ai_4/(li_0+ai_1+ai_2+ai_3)
    ci5 = ai_5/(li_0+ai_1+ai_2+ai_3+ai_4)

    d1 = ctf(2/VA)
    d2 = ctf(4/VA)
    d3 = ctf(8/VA)
    d4 = ctf(16/VA)
    d5 = ctf(32/VA)

    ai_1 = cmaskn(c1, ci1, a_1, ai_1, 1)
    ai_2 = cmaskn(c2, ci2, a_2, ai_2, 2)
    ai_3 = cmaskn(c3, ci3, a_3, ai_3, 3)
    ai_4 = cmaskn(c4, ci4, a_4, ai_4, 4)
    ai_5 = cmaskn(c5, ci5, a_5, ai_5, 5)

    l0 = l_0
    li0 = li_0
    a1 = gthresh(c1, d1, a_1)
    ai1 = gthresh(ci1, d1, ai_1)
    a2 = gthresh(c2, d2, a_2)
    ai2 = gthresh(ci2, d2, ai_2)
    a3 = gthresh(c3, d3, a_3)
    ai3 = gthresh(ci3, d3, ai_3)
    a4 = gthresh(c4, d4, a_4)
    ai4 = gthresh(ci4, d4, ai_4)
    a5 = gthresh(c5, d5, a_5)
    ai5 = gthresh(ci5, d5, ai_5)

    Os = l0+a1+a2+a3+a4+a5
    Is = li0+ai1+ai2+ai3+ai4+ai5

    A = np.sum(Os**2)
    square_err = (Os-Is)*(Os-Is)
    B = np.sum(square_err)
    nqm_value = 10*np.log10(A/B)
    return nqm_value


def extract_feature(img):
    row, col = img.shape
    img_feature = np.zeros([row, col, 4])
    
    # first order gradient filters
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    img_feature[:,:,0] = convolve2d(img, hf1, 'same')
    img_feature[:,:,1] = convolve2d(img, vf1, 'same')
    
    # Second order gradient filters
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T
    img_feature[:,:,2] = convolve2d(img, hf2, 'same')
    img_feature[:,:,3] = convolve2d(img, vf2, 'same')
    
    return img_feature

def extract_lr_feat(img_lr):
    """
    WRONG IMPLEMENTATION
    """
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat

def create_list_step(start, stop, step):
    list_step = []
    for i in range(start, stop, step):
        list_step = np.append(list_step, i)
    return list_step

def lin_scale(h_img, l_norm):
    h_norm = np.sqrt(np.sum(h_img*h_img))
    if h_norm>0:
        s = 1.2*l_norm/h_norm #? s = 1.2*l_norm/h_norm
        h_img = h_img*s
    return h_img

def scsr(img_lr_y, upscale_factor, Dh, Dl, lmbd, overlap, maxiter):
    # sparse coding super resolution
    
    # normalize the dictionary
    Dl = normalize(Dl,axis=0) #? normalize?  
    patch_size = int(np.sqrt(Dh.shape[0]))
       
    # bicubic interpolation of the lr image
    img_lr_y_upscale = resize(img_lr_y, np.multiply(upscale_factor, img_lr_y.shape), 3, preserve_range = True)
    
    img_sr_y_height,img_sr_y_width = img_lr_y_upscale.shape
    img_sr_y = np.zeros(img_lr_y_upscale.shape)
    cnt_matrix = np.zeros(img_lr_y_upscale.shape)
    
    # extract lr image features
    img_lr_y_feature = extract_feature(img_lr_y_upscale)
    
    # patch indexes for sparse recovery
    # drop 2 pixels at the boundary
    gridx = np.arange(3,img_sr_y_width-patch_size-2,patch_size-overlap)
    gridx = np.append(gridx,img_sr_y_width-patch_size-2)
    gridy = np.arange(3,img_sr_y_height-patch_size-2,patch_size-overlap)
    gridy = np.append(gridy,img_sr_y_height-patch_size-2)
    
    A = np.dot(Dl.T,Dl)

    # loop to recover each low-resolution patch
    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])
            
            patch = img_lr_y_upscale[yy:yy+patch_size, xx:xx+patch_size]
            patch_mean = np.mean(patch)
            patch = np.ravel(patch,'F') - patch_mean
            patch_norm = np.sqrt(np.sum(patch*patch))
            
            feature = img_lr_y_feature[yy:yy+patch_size, xx:xx+patch_size, :]
            feature = np.ravel(feature,'F')
            feature_norm = np.sqrt(np.sum(feature*feature))
            
            if feature_norm>1:
                feature = feature/feature_norm
            
            y = feature
                
            b = np.zeros([1,Dl.shape[1]])-np.dot(Dl.T,y)
            b = b.T

            # sparse recovery
            w = sparse_solution(lmbd, A, b, maxiter)
            
            # generate hr patch and scale the contrast
            h_patch = np.dot(Dh,w)
            
            # h_patch = np.zeros(patch.shape)
            h_patch = lin_scale(h_patch, patch_norm)
            
            h_patch = np.reshape(h_patch,[patch_size,patch_size])
            h_patch = h_patch+patch_mean
            
            img_sr_y[yy:yy+patch_size, xx:xx+patch_size] += h_patch
            cnt_matrix[yy:yy+patch_size, xx:xx+patch_size] += 1

    idx = np.where(cnt_matrix < 1)[0]
    img_sr_y[idx] = img_lr_y_upscale[idx]

    cnt_matrix[idx] = 1
    img_sr_y = img_sr_y/cnt_matrix
    
    return img_sr_y

def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap):

    patch_size = 3

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_hr)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0

    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            b = np.dot(np.multiply(Dl.T, -1), y)
            w = fss_yang(lmbd, Dl, b)

            hr_patch = np.dot(Dh, w)
            hr_patch = lin_scale(hr_patch, us_norm)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] += 1

    index = np.where(cnt_matrix < 1)[0]
    img_hr[index] = img_us[index]

    cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)

    return img_hr


if __name__ == "__main__":
    img = np.random.rand(5,5)
    # print(np.any(extract_feature(img) == extract_lr_feat(img)))

    print(lin_scale(img, 0))