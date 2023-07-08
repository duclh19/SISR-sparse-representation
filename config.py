

class TrainConfig(object): 
    train_path      = 'data/T91/'
    # hr_eval_path    = 'data/val_hr/'
    # lr_eval_path    = 'data/val_lr/'
    dictionary_path = 'dicts/'
    
    dict_size   = 1024
    lmbda       = 0.1 
    patch_size  = 3
    num_samples = 100000
    upscale     = 2
    prune_perc  = 10 # per 100
    max_iter    = 100

    hr_dict     = 'Dh_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'
    
    lr_dict     = 'Dl_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'

class EvalConfig(object): 
    hr_eval_path    = 'data/val_hr'
    lr_eval_path    = 'data/val_lr'
    dictionary_path = 'dicts/'

    overlap = 1
    lmbda   = 0.1
    upscale = 3
    color_space = 'ycbcr'
    max_iter = 100
    nu = 1
    beta = 0
    dict_size = 1024
    patch_size = 3
    hr_dict     = 'Dh_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'
    
    lr_dict     = 'Dl_' + str(dict_size) \
                + '_US' + str(upscale) \
                + '_L' + str(lmbda) \
                + '_PS' + str(patch_size) + '.pkl'

class TestConfig(): 
    
    def __init__(self, image_path, 
                 upscale_factor=2, 
                 dict_size=1024, 
                 lmbda=0.1, 
                 patch_size=3, 
                 overlap=1, 
                 nu=1, 
                 beta=0, 
                 ): 
        self.image_path = image_path
        self.upscale = upscale_factor
        self.dict_size = dict_size
        self.lmbda = lmbda
        self.overlap = overlap
        self.nu = nu
        self.beta = beta 
        self.patch_size = patch_size
        
        self._get_dict()

    def _get_dict(self): 
        self.hr_dict    = 'dicts/Dh_' + str(self.dict_size) \
                        + '_US' + str(self.upscale) \
                        + '_L' + str(self.lmbda) \
                        + '_PS' + str(self.patch_size) + '.pkl'
            
        self.lr_dict    = 'dicts/Dl_' + str(self.dict_size) \
                        + '_US' + str(self.upscale) \
                        + '_L' + str(self.lmbda) \
                        + '_PS' + str(self.patch_size) + '.pkl'
        
if __name__ == '__main__': 
    params = TrainConfig()
    print(params.hr_dict)
