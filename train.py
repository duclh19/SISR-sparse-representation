"""
For training dictionaries
"""
import os
import numpy as np
from requests import patch
from patches_proc import random_sample_patch, patch_pruning
from spams import trainDL
import pickle
from tqdm import tqdm



## PARAMETERS SET UP
dict_size = 512
lmbd = 0.1
patch_size = 3
num_samples = 100000
upscale = 2
prune_perc = 10 # percentage for prunning
iter = 100
train_img_path = './data/T91/'

dict_path = './dict/'
if not os.path.isdir(dict_path): 
    os.mkdir(dict_path)

assert os.path.exists(train_img_path), "Your path does not exist"
assert train_img_path.endswith('/'), 'Your path should end with / (slash)'


Xh, Xl = random_sample_patch(train_img_path, patch_size, num_samples, upscale)

Xh, Xl = patch_pruning(Xh, Xl, prune_perc)

Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

# DICTIONARY LEARNING
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)


# SAVE DICTIONARY
with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__": 
    test = np.asfortranarray(np.ones((2, 2))) 
    print(test)

