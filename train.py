
import numpy as np
# from utils import random_sample_patch, patch_pruning
from rnd_smp_patch import rnd_smp_patch, patch_pruning
from spams import trainDL
import pickle



class args(object):
    dict_size = 512
    lmbd = 0.1
    patch_size = 5
    nSmp = 100000
    upscale = 2
    prune_per = 10
    iter = 100
    train_img_path = './data/T91/'


para = args()

# Randomly sample image patches
Xh, Xl = random_sample_patch(
    para.train_img_path, para.patch_size, para.nSmp, para.upscale)
# Prune patches with small variances
Xh, Xl = patch_pruning(Xh, Xl, para.prune_per)

hDim = Xh.shape[0]
lDim = Xl.shape[0]

hNorm = np.sqrt(np.sum(Xh*Xh, axis=0))
lNorm = np.sqrt(np.sum(Xl*Xl, axis=0))

Idx = np.where((hNorm*lNorm) != 0)[0]

Xh = Xh[:, Idx]
Xl = Xl[:, Idx]

Xh = Xh/np.tile(np.sqrt(np.sum(Xh*Xh, axis=0)), [Xh.shape[0], 1])
Xl = Xl/np.tile(np.sqrt(np.sum(Xl*Xl, axis=0)), [Xl.shape[0], 1])

X = np.concatenate([Xh/np.sqrt(hDim), Xl/np.sqrt(lDim)], axis=0)
XNorm = np.sqrt(np.sum(X*X, axis=0))

X = X/np.tile(XNorm, [X.shape[0], 1])
X = np.asfortranarray(X)


D = trainDL(X, K=para.dict_size, lambda1=para.lmbd, iter=para.iter)
# %
Dh = D[0:hDim, :]*np.sqrt(hDim)
Dl = D[hDim:, :]*np.sqrt(lDim)
# %
with open('dictionary/' + 'Dh_' + str(para.dict_size) + '_US' + str(para.upscale) + '_L' + str(para.lmbd) + '_PS' + str(para.patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open('dictionary/' + 'Dl_' + str(para.dict_size) + '_US' + str(para.upscale) + '_L' + str(para.lmbd) + '_PS' + str(para.patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
