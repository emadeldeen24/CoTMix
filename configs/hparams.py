def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.9, 
                'temporal_shift': 14, 
                'src_cls_weight': 0.78,
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05
            }
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.79,
                'temporal_shift': 300, 
                'src_cls_weight': 0.96, 
                'src_supCon_weight': 0.1,
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05
            }
        }



class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001,
                'mix_ratio': 0.72,
                'temporal_shift': 14, 
                'src_cls_weight': 0.98,
                'src_supCon_weight': 0.1 , 
                'trg_cont_weight': 0.1,
                'trg_entropy_weight': 0.05,
            }
        }


class HHAR():
    def __init__(self):
        super(HHAR, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }

        self.alg_hparams = {
            'CoTMix': {
                'learning_rate': 0.001, 
                'mix_ratio': 0.52,
                'temporal_shift': 14,
                'src_cls_weight': 0.8,
                'src_supCon_weight': 0.1, 
                'trg_cont_weight': 0.1, 
                'trg_entropy_weight': 0.05, 
            }
        }


