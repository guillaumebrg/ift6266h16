__author__ = 'Guillaume'

import numpy as np
import platform
import os
import pickle
import time
import sys

from contextlib import contextmanager
from keras.callbacks import Callback, EarlyStopping
import keras.backend as K
from preprocessing import resize_pil, check_preprocessed_data, convert_labels, standardize_dataset, preprocess_dataset, \
    get_next_batch
from reporting import write_experiment_report, print_architecture
from training_params import TrainingParams
from dataset import InMemoryDataset, FuelDataset
from testing import get_best_model_from_exp, test_model, update_BN_params, generate_submission_file, \
    adapt_to_new_input, categorical_crossentropy, predict, test_ensemble_of_models, test_model_on_exp, generate_csv_file

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

def load_dataset_in_memory_and_resize(data_access, set, division, dataset_path, targets_path, tmp_size,
                                      final_size, batch_size):
    if data_access == "in-memory":
        with timer("Loading %s data"%set):
            dataset = InMemoryDataset(set, dataset_path, source_targets=targets_path, division=division)
            draw_data = np.copy(dataset.dataset)
            targets = np.copy(dataset.targets)
            del dataset
    elif data_access == "fuel":
        with timer("Loading %s data"%set):
            dataset = FuelDataset(set, tmp_size, batch_size=batch_size, shuffle=False, division=division)
            draw_data,targets = dataset.return_whole_dataset()
            del dataset
    else:
        raise Exception("Data access not available. Must be 'fuel' or 'in-memory'. Here : %s."%data_access)

    if tmp_size != final_size:
        # Resize images from the validset
        out = np.zeros((draw_data.shape[0], final_size[0], final_size[1], final_size[2]), dtype="float32")
        with timer("Resizing %s images"%set):
            for i in range(draw_data.shape[0]):
                out[i] = resize_pil(draw_data[i], final_size[0:2])
        del draw_data
        return out, targets
    else:
        return draw_data, targets

def launch_training(training_params):
    """
    Load the data, and train a Keras model.

    :param training_params: a TrainingParams object which contains each parameter of the training
    :return:
    """
    if os.path.exists(training_params.path_out) is False:
        os.mkdir(os.path.abspath(training_params.path_out))

    ###### LOADING VALIDATION DATA #######
    validset, valid_targets = load_dataset_in_memory_and_resize(training_params.data_access, "valid",
                                                                training_params.division, training_params.dataset_path,
                                                                training_params.targets_path, training_params.final_size,
                                                                training_params.final_size, training_params.test_batch_size)
    valid_targets = convert_labels(valid_targets)

    ###### Preprocessing VALIDATION DATA #######
    for mode in training_params.valid_preprocessing:
        validset = preprocess_dataset(validset, training_params, mode)
    # Transpose validset >> (N, channel, X, Y)
    validset = validset.transpose(0,3,1,2)
    # Multiple input ?
    if training_params.multiple_inputs>1:
        validset = [validset for i in range(training_params.multiple_inputs)]

    ###### MODEL INITIALIZATION #######
    with timer("Model initialization"):
        model = training_params.initialize_model()
    if training_params.pretrained_model is not None:
        with timer("Pretrained Model initialization"):
            pretrained_model = training_params.initialize_pretrained_model()
            training_params.generator_args.append(pretrained_model)
            # preprocessed the validset
            if type(pretrained_model) is list:
                features = []
                for pmodel in pretrained_model:
                    features.append(pmodel.predict(validset))
                validset = np.concatenate(features, axis=1)
            else:
                validset = pretrained_model.predict(validset)

    ###### SAVE PARAMS ######
    s = training_params.print_params()
    # Save command
    f = open(training_params.path_out+"/command.txt", "w")
    f.writelines(" ".join(sys.argv))
    f.writelines(s)
    f.close()
    # Print architecture
    print_architecture(model, path_out=training_params.path_out + "/architecture.txt")

    ###### TRAINING LOOP #######
    count = training_params.fine_tuning

    with timer("Training"):
        while training_params.learning_rate >= training_params.learning_rate_min and count<training_params.nb_max_epoch:

            if count != 0: # Restart from the best model with a lower LR
                model = training_params.initialize_model()
                model.load_weights(training_params.path_out+"/MEM_%d/best_model.cnn"%(count-1))
            # Callbacks
            early_stoping = EarlyStopping(monitor="val_loss",patience=training_params.max_no_best)
            save_model = ModelCheckpoint_perso(filepath=training_params.path_out+"/MEM_%d"%count, verbose=1,
                                               optional_string=s, monitor="val_acc", mode="acc")

            history = model.fit_generator(training_params.generator(*training_params.generator_args),
                                          nb_epoch=training_params.nb_max_epoch,
                                          samples_per_epoch= int(training_params.Ntrain*training_params.bagging_size),
                                          show_accuracy=True,
                                          verbose=training_params.verbose,
                                          validation_data=(validset,  valid_targets),
                                          callbacks=[early_stoping, save_model])

            training_params.learning_rate *= 0.1
            training_params.update_model_args()
            save_history(training_params.path_out+"/MEM_%d/history.pkl"%count, history)
            count += 1

def launch_adversarial_training(training_params):
    """
    Load the data, and train a Keras model.

    :param training_params: a TrainingParams object which contains each parameter of the training
    :return:
    """
    if os.path.exists(training_params.path_out) is False:
        os.mkdir(os.path.abspath(training_params.path_out))

    ###### LOADING VALIDATION DATA #######
    validset, valid_targets = load_dataset_in_memory_and_resize(training_params.data_access, "valid", training_params.dataset_path,
                                                                training_params.targets_path, training_params.final_size,
                                                                training_params.final_size, training_params.test_batch_size)
    valid_targets = convert_labels(valid_targets)

    ###### Preprocessing VALIDATION DATA #######
    for mode in training_params.valid_preprocessing:
        validset = preprocess_dataset(validset, training_params, mode)
    # Transpose validset >> (N, channel, X, Y)
    validset = validset.transpose(0,3,1,2)
    # Multiple input ?
    if training_params.multiple_inputs>1:
        validset = [validset for i in range(training_params.multiple_inputs)]

    ###### MODEL INITIALIZATION #######
    with timer("Model initialization"):
        model = training_params.initialize_model()
    if training_params.pretrained_model is not None:
        with timer("Pretrained Model initialization"):
            pretrained_model = training_params.initialize_pretrained_model()
            training_params.generator_args.append(pretrained_model)
            # preprocessed the validset
            if type(pretrained_model) is list:
                features = []
                for pmodel in pretrained_model:
                    features.append(pmodel.predict(validset))
                validset = np.concatenate(features, axis=1)
            else:
                validset = pretrained_model.predict(validset)

    ###### SAVE PARAMS ######
    s = training_params.print_params()
    # Save command
    f = open(training_params.path_out+"/command.txt", "w")
    f.writelines(" ".join(sys.argv))
    f.writelines(s)
    f.close()
    # Print architecture
    print_architecture(model, path_out=training_params.path_out + "/architecture.txt")

    ###### TRAINING SET #######

    train_dataset = FuelDataset("train", training_params.tmp_size,
                                batch_size=training_params.batch_size,
                                bagging=training_params.bagging_size,
                                bagging_iterator=training_params.bagging_iterator)

    ###### ADVERSARIAL MAPPING ######

    input_ = model.layers[0].input
    y_ = model.y
    layer_output = model.layers[-1].get_output()
    xent = K.categorical_crossentropy(y_, layer_output)
    loss = xent.mean()
    grads = K.gradients(loss, input_)
    get_grads = K.function([input_, y_], [loss, grads])

    ###### TRAINING LOOP #######
    count = training_params.fine_tuning
    epoch_count = 0

    with timer("Training"):
        while training_params.learning_rate >= training_params.learning_rate_min and epoch_count<training_params.nb_max_epoch:

            if count != 0: # Restart from the best model with a lower LR
                model = training_params.initialize_model()
                model.load_weights(training_params.path_out+"/MEM_%d/best_model.cnn"%(count-1))
                # Recompile get_grads
                input_ = model.layers[0].input
                y_ = model.y
                layer_output = model.layers[-1].get_output()
                xent = K.categorical_crossentropy(y_, layer_output)
                loss = xent.mean()
                grads = K.gradients(loss, input_)
                get_grads = K.function([input_, y_], [loss, grads])

            best = 0.0
            patience = training_params.max_no_best
            losses = []
            adv_losses = []
            accuracies = []
            adv_accuracies = []
            valid_losses = []
            valid_accuracies = []
            epoch_count = 0
            no_best_count = 0
            path = training_params.path_out + "/MEM_%d"%count
            if os.path.exists(path) is False:
                os.mkdir(path)
            # Log file
            f = open(path+"/log.txt", "w")
            f.write("LR = %.2f\n"%training_params.learning_rate)
            f.close()
            # Config file
            open(path+"/config.netconf", 'w').write(model.to_json())

            while no_best_count < patience and epoch_count < training_params.nb_max_epoch:
                new = True
                loss = 0.0
                adv_loss = 0.0
                accuracy = 0.0
                adv_accuracy = 0.0
                # Trainset Loop
                N = training_params.Ntrain/(training_params.batch_size*1)
                for i in range(N):
                    # Train
                    print "\rEpoch %d : Batch %d over %d"%(epoch_count, i, N),
                    processed_batch, labels = get_next_batch(train_dataset, training_params.batch_size,
                                                             training_params.final_size,
                                                             training_params.preprocessing_func,
                                                             training_params.preprocessing_args)
                    l, acc = model.train_on_batch(processed_batch, labels, accuracy=True)
                    # Update stats
                    if new:
                        loss = l
                        accuracy = acc
                    else:
                        loss = 0.9*loss + 0.1*l
                        accuracy = 0.9*accuracy + 0.1*acc
                    # Get adversarial examples
                    l, grads = get_grads([processed_batch, labels])
                    updates = np.sign(grads)
                    adversarials = processed_batch + updates
                    # Train on adv examples
                    adv_l, adv_acc = model.train_on_batch(adversarials, labels, accuracy=True)
                    # Update stats
                    if new:
                        adv_loss = adv_l
                        adv_accuracy = adv_acc
                        new = False
                    else:
                        adv_loss = 0.9*adv_loss + 0.1*adv_l
                        adv_accuracy = 0.9*adv_accuracy + 0.1*adv_acc
                # Store stats
                losses.append(loss)
                accuracies.append(accuracy)
                adv_losses.append(adv_loss)
                adv_accuracies.append(adv_accuracy)
                # Validset loss and accuracy
                out = model.predict(validset)
                valid_loss = categorical_crossentropy(valid_targets, out)
                count = np.sum(np.argmax(valid_targets, axis=1) - np.argmax(out, axis=1) == 0)
                score = float(count)/valid_targets.shape[0]
                valid_losses.append(valid_loss)
                valid_accuracies.append(score)

                # Stop criterion and Save model
                string = "***\nEpoch %d: Loss : %0.5f, Adv loss : %0.5f, Valid loss : %0.5f, " \
                         "Acc : %0.5f, Adv acc : %0.5f, Valid acc : %0.5f"%(epoch_count, losses[-1], adv_losses[-1],
                                                                            valid_losses[-1], accuracies[-1],
                                                                            adv_accuracies[-1], valid_accuracies[-1])
                if score > best:
                    no_best_count = 0
                    save_path = path+"/best_model.cnn"
                    if training_params.verbose>0:
                        string = string +"\tBEST\n"
                        print string
                        write_log(path+"/log.txt", string)
                    best = score
                    model.save_weights(save_path, overwrite=True)
                else:
                    no_best_count += 1
                    save_path = path+"/last_epoch.cnn"
                    if training_params.verbose>0:
                        string = string + "\n"
                        print string
                        write_log(path+"/log.txt", string)
                    model.save_weights(save_path, overwrite=True)
                epoch_count += 1

            # Update learning rate
            training_params.learning_rate *= 0.1
            training_params.update_model_args()
            with open(path + "/history.pkl","w") as f:
                pickle.dump(losses,f)
                pickle.dump(adv_losses,f)
                pickle.dump(valid_losses,f)
                pickle.dump(accuracies,f)
                pickle.dump(adv_accuracies,f)
                pickle.dump(valid_accuracies,f)
            count += 1


if __name__ == "__main__":
    end = False
    mode = ""
    try:
        mode = sys.argv[1]
    except:
        print "Expects an argument : '-train', '-check', '-report', '-test' or 'ensemble'."
        end = True

    if end is not True:
        training_params = TrainingParams()
        if mode=="-train":
            for i in range(training_params.multiple_training):
                launch_training(training_params)
                if training_params.division == "leaderboard":
                    generate_submission_file(training_params)
                else:
                    test_model_on_exp(training_params, verbose=True, write_txt_file=True)
                if platform.system()=="Windows":
                    write_experiment_report(training_params.path_out, multipages=True)
                    write_experiment_report(training_params.path_out, multipages=False)
                training_params.update_params_for_next_training()
        if mode=="-adversarial":
            for i in range(training_params.multiple_training):
                launch_adversarial_training(training_params)
                test_model_on_exp(training_params, verbose=True, write_txt_file=True)
                if platform.system()=="Windows":
                    write_experiment_report(training_params.path_out, multipages=True)
                    write_experiment_report(training_params.path_out, multipages=False)
                training_params.update_params_for_next_training()
        elif mode=="-check":
            try:
                n = int(sys.argv[2])
            except:
                n=10
            check_preprocessed_data(training_params.data_access,
                                    training_params.dataset_path,
                                    training_params.targets_path,
                                    training_params.batch_size,
                                    training_params.tmp_size,
                                    training_params.final_size,
                                    training_params.preprocessing_func,
                                    training_params.preprocessing_args,
                                    n=n)
        elif mode=="-report":
            write_experiment_report(training_params.path_out, multipages=True)
            write_experiment_report(training_params.path_out, multipages=False)
        elif mode=="-test":
            test_model_on_exp(training_params, verbose=True, write_txt_file=True)
        elif mode=="-ensemble":
            test_ensemble_of_models(training_params)
        elif mode=="-submit":
            generate_submission_file(training_params)
        elif mode == "-generate_fucking_features_for_antoine":
            generate_csv_file("final_preds", -1, training_params)
        else:
            print "Mode not undertstood. Use '-train', '-check', '-report', or '-test'. Here : %s"%mode
