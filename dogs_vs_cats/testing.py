__author__ = 'Guillaume'

import numpy as np
from keras.models import model_from_json, model_from_config, Sequential
from keras import backend as K
import os
import pickle
from scipy.ndimage.filters import gaussian_filter

def categorical_crossentropy(ytrue, ypred, eps=1e-6):
    return -np.mean((ytrue*np.log(ypred+eps)).sum(axis=1))

def get_loss_and_acc(path_to_experiment, subdir_names="MEM", history_files_name="history.pkl", return_dirs=False):
    expe_list = os.listdir(path_to_experiment)
    expe_list = [path_to_experiment+"/"+dir_ for dir_ in expe_list if dir_.find(subdir_names)==0]
    # Open history.pkl files
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    for dir_ in expe_list:
        with open(dir_+"/%s"%history_files_name, "r") as f:
            train_loss.append(pickle.load(f))
            valid_loss.append(pickle.load(f))
            train_acc.append(pickle.load(f))
            valid_acc.append(pickle.load(f))
    if return_dirs:
        return train_loss, valid_loss, train_acc, valid_acc, expe_list
    return train_loss, valid_loss, train_acc, valid_acc

def get_model(path, weights_name="best_model.cnn", config_name="config.netconf"):
     # Path to the weights and config
    best_model_weights = os.path.join(path, weights_name)
    best_model_config =os.path.join(path, config_name)
    # Create the model from cofig
    model = model_from_json(open(best_model_config).read())
    # Load weights
    model.load_weights(best_model_weights)
    return model, best_model_weights

def get_best_model_from_exp(path_to_experiment, subdir_names="MEM", history_files_name="history.pkl",
                            weights_name="best_model.cnn", config_name="config.netconf"):

    train_loss, valid_loss, train_acc, valid_acc, expe_list = get_loss_and_acc(path_to_experiment, subdir_names,
                                                                               history_files_name, return_dirs=True)
    # Get the best model on the validation set
    best_index = np.argmax([max(acc) for acc in valid_acc])
    # Create the model from cofig
    model, path_model = get_model(expe_list[best_index], weights_name, config_name)
    return model, path_model


def test_model(model, dataset, labels, batch_size=32, return_preds=False, verbose=True):
    """
    Expects a 4D dataset (N, rows, cols, channels).
    Expects binary labels.
    """
    predictions = model.predict(np.array(dataset,"float32"), batch_size=batch_size)
    test_loss = categorical_crossentropy(labels, predictions).mean()
    count = np.sum(np.argmax(labels, axis=1) - np.argmax(predictions, axis=1) == 0)
    score = float(count)/labels.shape[0]
    if verbose:
        print "Accuracy = %.3f"%(score)
        print "Test Loss = %.3f"%(test_loss)
    if return_preds:
        return score, test_loss, predictions
    else:
        return score, test_loss

def update_BN_params(model, dataset, batch_size, scale=1.0, N=100, verbose=False):
    # Get the names of each layer
    names_layers = [l.name for l in model.layers]
    # Find BN layers
    check_BN = np.array([name.find('batchnormalization') for name in names_layers])
    BN_positions = np.argwhere(check_BN>-1)
    BN_positions = list(BN_positions.flatten())
    print BN_positions
    if verbose:
        print "%d BN layers found."%len(BN_positions)
    # If no BN, return
    if not BN_positions:
        return
    for j,pos in enumerate(BN_positions):
        if verbose:
            print "\rUpdating BN #%d"%(j+1)
        # Get the BN layer and the previous one
        previous_BN_layer = model.layers[pos-1]
        BN_layer = model.layers[pos]
        # Get current weights (shape = [gamma, beta, mean, std])
        weights = BN_layer.get_weights()
        # Create a function able to get intermediate results
        intermediate_output = K.function([model.layers[0].input],
                                  [previous_BN_layer.get_output(train=False)])
        # Get the shape of mean and std (taken from Keras Batch normalization code)
        input_shape = BN_layer.input_shape
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[BN_layer.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[BN_layer.axis] = input_shape[BN_layer.axis]
        # Initiate new mean and std
        new_mean = np.zeros(broadcast_shape, "float32").flatten()
        new_std = np.zeros(broadcast_shape, "float32").flatten()
        # First, shuffle the dataset
        count = 0.0
        # Then, compute new mean and new std
        for i in range(N):
            # Batch processing (taken from Keras Batch normalization code)
            batch = dataset.get_batch()[0]/scale
            out = intermediate_output([batch.transpose(0,3,1,2)])[0]
            m = mean_with_list_axis(out, reduction_axes)
            std = mean_with_list_axis(np.square(out-m.reshape(broadcast_shape)) + BN_layer.epsilon, reduction_axes)
            std = np.sqrt(std)
            # Statistics accumulation
            new_mean += m
            new_std += std
            count += 1.0
            if verbose:
                print "\r%d processed examples..."%(batch_size*i),
        # Averaging
        new_mean /= count
        new_std = new_std/count # Adding eps in order to avoid 0 division
        # Setting new weights
        print new_mean[0]
        print weights[2][0]
        weights[2]=new_mean
        print new_std[0]
        print weights[3][0]
        weights[3]=new_std
        BN_layer.set_weights(np.array(weights, "float32"))
    if verbose:
        print "\rDone."

def softmax(input_):
    tmp = np.exp(input_)
    return tmp/tmp.sum(axis=1)

def return_detection_model(model, new_shape):
    """
    This function adapt input shape of the model.
    If there is "Flatten layer" it will remove every above layers.
    """
    # Get the config of the trained model
    config_dict = model.get_config()
    # Adapt input shape
    config_dict['layers'][0]['input_shape'] = new_shape
    # Check for "Flatten + Activation" :
    # Get the names of each layer
    names_layers = [l.name for l in model.layers]
    # Find BN layers
    check_flatten = np.array([name.find('flatten') for name in names_layers])
    flatten_position = np.argwhere(check_flatten==0)
    if flatten_position:
        config_dict['layers']=config_dict['layers'][0:flatten_position[0]]
    # Create a new model from this config
    model_detect = model_from_config(config_dict)
    # Set weights
    for i,layer in enumerate(model_detect.layers):
        layer.set_weights(model.layers[i].get_weights())
    return model_detect

def apply_detection(imgs, model_detect, activation="softmax", blur=True, sigma=1.5):
    # Get model output
    preds = model_detect.predict(imgs.transpose(0,3,1,2))
    # Apply softmax ?
    if activation=="softmax":
        preds = softmax(preds)
    if blur:
        for i,pred in enumerate(preds):
            for j,predmap in enumerate(pred):
                preds[i,j] = gaussian_filter(predmap, sigma)
    return preds


def mean_with_list_axis(a, ax):
    out = np.mean(a, axis=ax[-1])
    if len(ax)>1:
        for i in ax[::-1][1:]:
            out = np.mean(out, axis=i)
    return out

