__author__ = 'Guillaume'

import def_model.vggnet_with_BN_window_46 as net
import pretrained_models.w46_w100_merged_multi as merge_model
import os
import platform
from preprocessing import images_generator, rotate_crop_and_scale, rotate_crop_and_standardize, \
    rotate_crop_and_mean, features_generator, multi_features_generator

class TrainingParams():
    def __init__(self):
        # Model definition
        self.model_definition = net
        self.model_args = []
        # Training parameters
        self.learning_rate = 0.08
        self.learning_rate_min = 0.0008
        self.momentum = 0.9
        self.max_no_best = 10
        self.nb_max_epoch = 1000
        self.verbose = 2
        self.batch_size = 32
        # Data processing
        self.Ntrain = 17500 # Only used to set the nb of examples per epoch in the fit_generator function
        self.Nvalid = 3750 # Not used
        self.Ntest = 3750 # Not used
        self.bagging_iterator = 0
        self.bagging_size = 1.0
        self.tmp_size = (250,250,3)
        self.final_size = (150,150,3)
        self.scale = 1.0
        self.blur = None # float or None
        self.rgb_alterate = False
        self.max_rotation = 10
        self.max_crop_rate = 5
        self.pretrained_model = None # used if preprocessing = 'features_generator', else set to None
        self.data_access = "fuel"
        self.division = "not_leaderboard" # 'leaderboard' to use the dataset split proposed on the leaderboard page
        if platform.system()=="Linux":
            self.data_access = "fuel"
        self.dataset_path = "path_to_numpy_file" # only used if data_access = 'in-memory'
        self.targets_path = "path_to_numpy_file" # only used if data_access = 'in-memory'
        self.multiple_inputs = 1
        self.preprocessing_func = rotate_crop_and_scale
        self.preprocessing_args = [self.final_size[0:2], self.max_rotation, self.max_crop_rate, self.scale, self.blur,
                                    self.rgb_alterate] # add scale in the case or rotate_crop_and_scale (before self.blur)
        self.generator = images_generator
        self.generator_args = [self.data_access,
                               self.dataset_path,
                               self.targets_path,
                               self.batch_size,
                               self.tmp_size,
                               self.final_size,
                               self.bagging_size,
                               self.bagging_iterator,
                               self.multiple_inputs,
                               self.division,
                               self.preprocessing_func,
                               self.preprocessing_args]

        self.valid_preprocessing = ["scale"] # scale, std, mean, or blur
        # Saving dir
        self.path_out = os.path.abspath("experiments/blog_post_5_scale_invariant/vggnet_with_BN_RGB_window_100_v2")
        self.multiple_training = 1 # number of training - see update_params_for_next_training
        self.fine_tuning = 0 # If weights to finetune are in MEM_4, set self.fine_tuning = 5
        # Testing parameters
        self.test_sizes = [(270,270,3),(210,210,3),(150,150,3)]
        self.test_batch_size = 50
        # self.ensemble_models = ["experiments/blog_post_5_scale_invariant/vggnet_with_BN_RGB_window_46_v2_512",
        #                         "experiments/blog_post_5_scale_invariant/vggnet_with_BN_RGB_window_100_v2"]
        self.ensemble_models = [self.path_out]
        # Update arguments of the initialize the model
        self.update_model_args()

    def update_params_for_next_training(self):
        # Function called between each training
        self.learning_rate = 0.08
        self.preprocessing_args[8] = rotate_crop_and_scale
        self.preprocessing_args[9] = [(150,150), 10, 5, 1.0]
        self.valid_preprocessing = "scale"
        self.path_out = os.path.abspath("experiments/debug")


    def wrapper(self, func, args):
        return func(*args)

    def initialize_model(self):
        return self.wrapper(self.model_definition.define_model, self.model_args)

    def initialize_pretrained_model(self):
        return self.wrapper(self.pretrained_model.define_pretrained_model, self.model_args)

    def update_model_args(self):
        self.model_args = []
        self.model_args.append(self.learning_rate)
        self.model_args.append(self.momentum)

    def print_params(self):
        s = "Data processing :"
        s += "\n\t Ntrain : %d, Nvalid : %d"%(self.Ntrain, self.Nvalid)
        s += "\n\t Bagging : %.2f, %d"%(self.bagging_size, self.bagging_iterator)
        s += "\n\t TMP size : (%d,%d)"%(self.tmp_size[0], self.tmp_size[1])
        s += "\n\t Final size : (%d,%d)"%(self.final_size[0], self.final_size[1])
        s += "\n\t Preprocessing function : %s"%self.preprocessing_func.__name__
        args = ""
        for elem in self.preprocessing_args:
            try:
                args += "%s, "%(str(elem))
            except:
                args += "???, "
        s += "\n\t with args : %s"%args
        if self.data_access=="in-memory":
            s += "\n\t Dataset : %s"%self.dataset_path
            s += "\n\t Targets : %s"%self.targets_path
        else:
            s += "\n\t Data access : %s"%self.data_access
        s += "\n\nTraining parameters :"
        s += "\n\t Learning rate : %.5f"%(self.learning_rate)
        s += "\n\t Learning rate lower bound : %.5f"%(self.learning_rate_min)
        s += "\n\t Momentum : %.5f"%(self.momentum)
        s += "\n\t Early stopping patience : %d"%(self.max_no_best)
        s += "\n\t Batch Size : %d"%(self.batch_size)
        s += "\n\nOther :"
        s += "\n\t Multiple Training : %d"%(self.multiple_training)
        s += "\n\t Fine-tuning : %d"%(self.fine_tuning)

        print s
        return s






