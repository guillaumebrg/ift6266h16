__author__ = 'Guillaume'

import def_model.vggnet_with_dropout as vggnet_with_dropout

from preprocessing import generator_from_file, rotate_crop_and_scale, resize_and_scale

class TrainingParams():
    def __init__(self):
        # Model definition
        self.model_definition = vggnet_with_dropout
        self.model_args = []
        # Data processing
        self.Ntrain = 20000
        self.Nvalid = 5000
        self.final_size = (150,150,1)
        self.preprocessing = generator_from_file
        self.preprocessing_args = ["data\\grayscale_250x250\\trainset_uint8_250x250x1.npy",
                                   "data\\grayscale_250x250\\trainset_targets_uint8_250x250x1.npy",
                                   (32, 1, 150, 150),
                                   rotate_crop_and_scale,
                                   [(150,150), 10, 5, 255.0]]
        # self.preprocessing_args = [["data\\rgb_200x200\\trainset_uint8_200x200x3_part%d.npy"%count for count in range(4)],
        #                            ["data\\rgb_200x200\\trainset_targets_uint8_200x200x3_part%d.npy"%count for count in range(4)],
        #                            (32, 3, 100, 100),
        #                            rotate_crop_and_scale,
        #                            [(100,100), 10, 5, 255.0]]
        self.validset = "data\\grayscale_250x250\\validset_uint8_250x250x1.npy"
        self.valid_targets = "data\\grayscale_250x250\\validset_targets_uint8_250x250x1.npy"
        # Training parameters
        self.learning_rate = 0.01
        self.learning_rate_min = 0.0001
        self.momentum = 0.2
        self.max_no_best = 10
        self.nb_max_epoch = 1000
        self.verbose = 2
        self.batch_size = 32
        # Saving dir
        self.path_out = "experiments\\vggnet_150x150_dropout_first_layers"

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






