# SISR-sparse-representation
Implementation of "Single Image Super-Resolution via Sparse Representation" by Yang et al. for educational purpose.


1. What inside folder `data/`, `dicts/`, and `results`

```
data/
  
│   ├── T91/    # train set 
│   ├── val_hr  # high-resolution val set 
│   │   ├── # Set 5 original
│   ├── val_lr  # low_resolution val set.
│   │   ├── # Set 5corresponding
```

```
dict/
│   ├── # all dictionaries trained (with main.py)
``` 

```
result/ 
│   ├── # the output for inference here
```

> The dataset can be download [here](https://drive.google.com/drive/folders/15PHLMjOuhZdffTkHqqyzWr4UNbDx8axf?usp=sharing)


[`train.py`](train.py): use to train dictionaries

[`main.py`](main.py)  : re-experiment on validation set. 

[`inference.py`](inference.py): test your own image. 

[`utils.py`](utils.py) and [`patches_proc.py`](patches_proc.py): for utils

`proc.py`: pre-preprocessing to generate low-res images (not in use anymore)