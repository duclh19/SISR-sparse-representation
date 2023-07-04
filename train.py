"""
For training dictionaries
"""
# import imp
import os
import numpy as np
from requests import patch
from patches_proc import random_sample_patch, patch_pruning
from spams import trainDL
import pickle
from tqdm import tqdm
from config import TrainConfig


params = TrainConfig()

if not os.path.isdir(params.dictionary_path): 
    os.mkdir(params.dictionary_path)
    print(f'\tCreate folder containing dictionaries: {params.dictionary_path}')
assert os.path.exists(params.train_path), "Your path does not exist"
assert params.train_path.endswith('/'), 'Your path should end with / (slash)'


Xh, Xl = random_sample_patch(params.train_path, params.patch_size, params.num_samples, params.upscale)

Xh, Xl = patch_pruning(Xh, Xl, params.prune_perc)

Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

# DICTIONARY LEARNING
Dh = trainDL(Xh, K=params.dict_size, lambda1=params.lmbda, iter=params.max_iter)
Dl = trainDL(Xl, K=params.dict_size, lambda1=params.lmbda, iter=params.max_iter)


# SAVE DICTIONARY
with open(os.path.join(params.dictionary_path, params.hr_dict), 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(params.dictionary_path, params.lr_dict), 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__": 
    # test = np.asfortranarray(np.ones((2, 2))) 
    # print(test)
    pass 

