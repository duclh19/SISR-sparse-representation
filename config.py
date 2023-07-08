

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
    dictionary_path = 'dicts/'

    overlap = 1
    lmbda   = 0.1
    upscale = 2
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
    # hr_img_path = 'data/test_flower.png'
    # lr_img_path = 'data/lr_test_flower.png'
    # img_name    = 'flower'
    def __init__(self, hr_img_path, lr_img_path, img_name=None) -> None:
        self.hr_img_path = hr_img_path
        self.lr_img_path = lr_img_path
        self.img_name    = img_name

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
