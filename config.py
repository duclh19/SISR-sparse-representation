

class TrainConfig(object): 
    train_path      = 'data/T91/'
    # hr_eval_path    = 'data/val_hr/'
    # lr_eval_path    = 'data/val_lr/'
    dictionary_path = 'dicts/'
    
    dict_size   = 2048
    lmbda       = 0.1 
    patch_size  = 3
    num_samples = 100000
    upscale     = 2
    prune_perc  = 10 # per 100
    max_iter    = 100

    hr_dict     = 'Dhr_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'
    
    lr_dict     = 'Dlr_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'

class EvalConfig(object): 
    hr_eval_path    = 'data/val_hr'
    lr_eval_path    = 'data/val_lr'
    dictionary_path = 'dicts/'

    overlap = 1
    lmbda   = 0.1 
    upscale = 2
    color_space = 'ycbcr'
    max_iter = 100
    nu = 1
    beta = 0
    dict_size = 2048
    patch_size = 3
    hr_dict     = 'Dhr_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'
    
    lr_dict     = 'Dlr_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'
if __name__ == '__main__': 
    params = TrainConfig()
    print(params.hr_dict)
