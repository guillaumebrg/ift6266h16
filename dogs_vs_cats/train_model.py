__author__ = 'Guillaume'

import json
import numpy as np
import os
import pickle
import time
import sys

from contextlib import contextmanager
from keras.callbacks import Callback, EarlyStopping
from preprocessing import resize, chech_preprocessed_data
from reporting import write_experiment_report
from training_params import TrainingParams

def save_history(path, history):
    """
    Save the loss, validation loss, accuracy and validation accuracy of a Keras training into a pickle file.

    :param path: where to save the pickle file
    :param history: an History object returned by the fit function of Keras
    :return:
    """
    with open(path,"w") as f:
        pickle.dump(history.history["loss"],f)
        pickle.dump(history.history["val_loss"],f)
        pickle.dump(history.history["acc"],f)
        pickle.dump(history.history["val_acc"],f)

class ModelCheckpoint_perso(Callback):
    """
    Keras callback subclass which defines a saving procedure of the model being trained : after each epoch,
    the last model is saved under the name 'after_random.cnn'. The best model is saved with the name 'best_model.cnn'.
    The model after random can also be saved. And the model architecture is saved with the name 'config.network'.
    Everything is stored using pickle.
    """
    def __init__(self, filepath, monitor='val_acc', verbose=1, save_best_only=False, save_first=True, optional_string="",
                 mode="acc"):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_first = save_first
        self.optional_string = optional_string
        self.mode = mode
        if mode == "acc":
            self.best = -np.Inf
        elif mode == "loss":
            self.best = np.Inf
        else:
            print "Mode not undestood. It should be 'loss' or 'acc'."


    def on_epoch_begin(self, epoch, logs={}):
        if epoch==0:
            if os.path.exists(self.filepath) is False:
                os.mkdir(self.filepath)

            open(self.filepath+"/config.netconf", 'w').write(self.model.to_json())

            save_path = self.filepath+"/after_random.cnn"
            if self.verbose > 0:
                f = open(self.filepath+"/log.txt", "w")
                f.write(self.optional_string)
                f.write("***\nEpoch %05d: %s after random  model saved to %s\n"%(epoch, self.monitor, save_path))
                f.close()
            if self.save_first:
                self.model.save_weights(save_path, overwrite=True)

    def on_epoch_end(self, epoch, logs={}):
        # SAVING WEIGHTS
        current = logs.get(self.monitor)
        if self.mode=="acc":
            condition = current > self.best
        else:
            condition = current < self.best
        if condition:
            save_path = self.filepath+"/best_model.cnn"
            if self.verbose > 0:
                string = "***\nEpoch %05d: %s improved from %0.5f to %0.5f\n"% (epoch, self.monitor, self.best, current)
                write_log(self.filepath+"/log.txt", string)
            self.best = current
            self.model.save_weights(save_path, overwrite=True)

        else:
            save_path = self.filepath+"/last_epoch.cnn"
            if self.verbose > 0:
                string = "***\nEpoch %05d: %s did not improve : %0.5f\n"% (epoch, self.monitor, current)
                write_log(self.filepath+"/log.txt", string)
            self.model.save_weights(save_path, overwrite=True)

def write_log(path, string):
    """
    Add a line at the end of a textfile.

    :param path: textfile location
    :param string: line to add
    """
    # Open and Read
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    # Adding a line
    lines.append(string)
    # Write
    f = open(path, "w")
    f.writelines(lines)
    f.close()

@contextmanager
def timer(name):
    """
    Print the time taken by some operations. Usage :
    with timer("Operation A"):
        operation

    :param name: just a string name for the print function
    :return:
    """
    start_time = time.time()
    yield
    stop_time = time.time()
    print('\n{} took {} seconds'.format(name, stop_time - start_time))

def launch_training(training_params):
    """
    Load the data, and train a Keras model.

    :param training_params: a TrainingParams object which contains each parameter of the training
    :return:
    """
    if os.path.exists(training_params.path_out) is False:
        os.mkdir(os.path.abspath(training_params.path_out))

    ###### LOADING DATA #######
    with timer("Loading validset data"):
        draw_validset = np.load(training_params.validset)
        valid_targets = np.load(training_params.valid_targets)
    # Resize images from the validset
    validset = np.zeros((training_params.Nvalid, training_params.final_size[2], training_params.final_size[0], training_params.final_size[1]),
                        dtype="float32")
    with timer("Resizing validset images"):
        for i in range(training_params.Nvalid):
            validset[i] = resize(draw_validset[i], training_params.final_size[0:2]).transpose(2,0,1)
    del draw_validset

    ###### MODEL INITIALIZATION #######
    with timer("Model initialization"):
        model = training_params.initialize_model()

    ###### SAVE PARAMS ######
    s = training_params.print_params()
    # Save command
    f = open(training_params.path_out+"/command.txt", "w")
    f.writelines(" ".join(sys.argv))
    f.writelines(s)
    f.close()

    ###### TRAINING LOOP #######
    count = 0

    with timer("Training"):
        while training_params.learning_rate >= training_params.learning_rate_min and count<training_params.nb_max_epoch:

            if count != 0: # Restart from the best model with a lower LR
                model = training_params.initialize_model()
                model.load_weights(training_params.path_out+"/MEM_%d/best_model.cnn"%(count-1))
            # Callbacks
            early_stoping = EarlyStopping(monitor="val_loss",patience=training_params.max_no_best)
            save_model = ModelCheckpoint_perso(filepath=training_params.path_out+"/MEM_%d"%count, verbose=1,
                                               optional_string=s, monitor="val_acc", mode="acc")

            history = model.fit_generator(training_params.preprocessing(*training_params.preprocessing_args),
                                          nb_epoch=training_params.nb_max_epoch,
                                          samples_per_epoch=20000,
                                          show_accuracy=True,
                                          verbose=training_params.verbose,
                                          validation_data=(validset/255.0,np.array(valid_targets, "float32")),
                                          callbacks=[early_stoping, save_model])

            training_params.learning_rate *= 0.1
            training_params.update_model_args()
            save_history(training_params.path_out+"/MEM_%d/history.pkl"%count, history)
            count += 1

if __name__ == "__main__":
    end = False
    mode = ""
    try:
        mode = sys.argv[1]
    except:
        print "Expects an argument : '-train' or '-check'"
        end = True

    if end is not True:
        training_params = TrainingParams()
        if mode=="-train":
            launch_training(training_params)
            write_experiment_report(training_params.path_out, multipages=True)
            write_experiment_report(training_params.path_out, multipages=False)
        elif mode=="-check":
            n = int(sys.argv[2])
            training_params.preprocessing_args.append(n)
            chech_preprocessed_data(*training_params.preprocessing_args)
        else:
            print "Mode not undertstood. '-train' or '-check' are available."
