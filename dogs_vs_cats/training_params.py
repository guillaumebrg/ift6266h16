__author__ = 'Guillaume'

import def_model.vggnet_with_regularisation as vggnet_with_regularisation
import os
import platform

from preprocessing import images_generator, rotate_crop_and_scale, resize_and_scale

class TrainingParams():
    def __init__(self):
        # Model definition
        self.model_definition = vggnet_with_regularisation
        self.model_args = []
        # Training parameters
        self.learning_rate = 0.01
        self.learning_rate_min = 0.0001
        self.momentum = 0.9
        self.max_no_best = 10
        self.nb_max_epoch = 1000
        self.verbose = 2
        self.batch_size = 32
        # Data processing
        self.Ntrain = 17500
        self.Nvalid = 3750
        self.Ntest = 3750
        self.tmp_size = (250,250,1)
        self.final_size = (150,150,1)
        self.scale = 255.0
        self.data_access = "fuel"
        self.dataset_path = "data/grayscale_250x250/dataset_uint8_250x250x1.npy"
        self.targets_path = "data/grayscale_250x250/targets.npy"
        self.preprocessing = images_generator
        self.preprocessing_args = [self.data_access,
                                   self.dataset_path,
                                   self.targets_path,
                                   self.batch_size,
                                   self.tmp_size,
                                   self.final_size,
                                   rotate_crop_and_scale,
                                   [(150,150), 10, 5, self.scale]]
        # Saving dir
        self.path_out = os.path.abspath("experiments/blog_post_3_regularization/vggnet_dropout_0.7")

        # Update arguments of the initialize the model
        self.update_model_args()
        if platform.system()=="Linux":
            self.data_access = "fuel"


    def wrapper(self, func, args):
        return func(*args)

    def initialize_model(self):
        return self.wrapper(self.model_definition.define_model, self.model_args)

    def update_model_args(self):
        self.model_args = []
        self.model_args.append(self.learning_rate)
        self.model_args.append(self.momentum)

    def print_params(self):
        s = "\nData processing :"
        s += "\n\t Ntrain : %d, Nvalid : %d"%(self.Ntrain, self.Nvalid)
        s += "\n\t Final size : (%d,%d)"%(self.final_size[0], self.final_size[1])
        s += "\n\t Preprocessing function : %s"%self.preprocessing.__name__
        args = ""
        for elem in self.preprocessing_args:
            try:
                args += "%s, "%(str(elem))
            except:
                args += "???, "
        s += "\n\t with args : %s"%args
        s += "\n\t Dataset : %s"%self.dataset_path
        s += "\n\t Targets : %s"%self.targets_path
        s += "\n\nTraining parameters :"
        s += "\n\t Learning rate : %.5f"%(self.learning_rate)
        s += "\n\t Learning rate lower bound : %.5f"%(self.learning_rate_min)
        s += "\n\t Momentum : %.5f"%(self.momentum)
        s += "\n\t Early stopping patience : %d"%(self.max_no_best)
        s += "\n\t Batch Size : %d"%(self.batch_size)
        s += "\n"

        print s
        return s






