__author__ = 'Guillaume'

import def_model.first_convnet as first_convnet

from preprocessing import generator_from_file, rotate_crop_and_scale, resize_and_scale

class TrainingParams():
    def __init__(self):
        # Model definition
        self.model_definition = first_convnet
        self.model_args = []
        # Data processing
        self.Ntrain = 20000
        self.Nvalid = 5000
        self.final_size = (100,100)
        self.preprocessing = generator_from_file
        self.preprocessing_args = ["data\\grayscale_200x200\\trainset_uint8_200x200.npy",
                                   "data\\grayscale_200x200\\trainset_targets_uint8_200x200.npy",
                                   32,
                                   rotate_crop_and_scale,
                                   [(100,100), 10, 10, 255.0]]
        self.validset = "data\\grayscale_200x200\\validset_uint8_200x200.npy"
        self.valid_targets = "data\\grayscale_200x200\\validset_targets_uint8_200x200.npy"
        # Training parameters
        self.learning_rate = 0.01
        self.learning_rate_min = 0.0001
        self.momentum = 0.2
        self.max_no_best = 10
        self.nb_max_epoch = 1000
        self.verbose = 2
        self.batch_size = 32
        # Saving dir
        self.path_out = "experiments\\first_cnn_with_data_aug"

        # Update arguments of the initialize the model
        self.update_model_args()

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
        s += "\n\t Validset location : %s"%self.validset
        s += "\n\t Valid targets location : %s"%self.valid_targets
        s += "\n\nTraining parameters :"
        s += "\n\t Learning rate : %.5f"%(self.learning_rate)
        s += "\n\t Learning rate lower bound : %.5f"%(self.learning_rate_min)
        s += "\n\t Momentum : %.5f"%(self.momentum)
        s += "\n\t Early stopping patience : %d"%(self.max_no_best)
        s += "\n\t Batch Size : %d"%(self.batch_size)
        s += "\n"

        print s
        return s






