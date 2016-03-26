__author__ = 'Guillaume'

import numpy as np
from keras.models import model_from_json, model_from_config
from keras import backend as K
from keras.layers.core import Flatten
from keras.models import Graph
import os
import pickle
from scipy.ndimage.filters import gaussian_filter
from dataset import InMemoryDataset, FuelDataset
from preprocessing import standardize_dataset, convert_labels, preprocess_dataset

def categorical_crossentropy(ytrue, ypred, eps=1e-6):
    return -np.mean((ytrue*np.log(ypred+eps)).sum(axis=1))

def update_BN_params(model, dataset, scale=1.0, N=100, verbose=False):
    # Get the names of each layer
    layers_name = [l.name for l in model.layers]
    # Find BN layers
    check_BN = np.array([name.find('batchnormalization') for name in layers_name])
    BN_positions = np.argwhere(check_BN>-1)
    BN_positions = list(BN_positions.flatten())
    # Get intermediate outputs
    intermediate_outputs = K.function([model.layers[0].input],
                                     [model.layers[bn-1].get_output(train=True) for bn in BN_positions])
    if verbose:
        print "%d BN layers found."%len(BN_positions)
    # If no BN, return
    if not BN_positions:
        return
    means = [0 for elem in BN_positions]
    stds = [0 for elem in BN_positions]
    # Compute statistics over N batches
    for i in range(N):
        if verbose:
            print "\rProcessing batch %d..."%(i),
        # Get next batch
        batch = dataset.get_batch()[0]/scale
        out = intermediate_outputs([batch.transpose(0,3,1,2)])
        for i,maps in enumerate(out):
            if maps.ndim == 4:
                maps = maps.transpose(1,0,2,3)
                shape = [maps.shape[0], np.prod(maps.shape[1:4])]
                maps = maps.reshape(shape)
                means[i] += np.mean(maps, axis=1)/N
                stds[i] += np.std(maps, axis=1)/N
            elif maps.ndim == 2:
                means[i] += np.mean(maps, axis=0)/N
                stds[i] += np.std(maps, axis=0)/N
    # Set new statistics
    for i,bn in enumerate(BN_positions):
        w = model.layers[bn].get_weights()
        w[2] = np.array(means[i], "float32")
        w[3] = np.array(stds[i], "float32")
        model.layers[bn].set_weights(w)

def softmax(input_):
    tmp = np.exp(input_)
    return tmp/tmp.sum(axis=1)

def adapt_to_new_input(model, input_shape, old_input_shape, verbose=False):
    old_config = model.get_config()
    config = old_config.copy()
    if config.has_key("nodes"):
        return adapt_graph_to_new_input(model, input_shape, old_input_shape, verbose)
    else:
        return adapt_model_to_new_input(model, input_shape, old_input_shape, verbose)

def adapt_model_to_new_input(model, input_shape, old_input_shape, verbose):
    old_config = model.get_config()
    config = old_config.copy()
    # Adapt input shape
    config['layers'][0]['input_shape']=input_shape
    # Adapt maxpool layer
    pool_size = [l.pool_size[0] for l in model.layers if l.name=="maxpooling2d"]
    global_stride = np.prod(pool_size[0:-1])
    new_pool_size = int(pool_size[-1]+(input_shape[1]-old_input_shape[1])/global_stride)
    if verbose:
        print "Input shape :", input_shape
        print "Poolsize :", pool_size[-1], "-> (%d,%d)"%(new_pool_size,new_pool_size)
    maxpool_pos = [i for i,l in enumerate(model.layers) if l.name=="maxpooling2d"]
    config['layers'][maxpool_pos[-1]]['pool_size']=(new_pool_size,new_pool_size)
    # Compile model
    new_model = model_from_config(config)
    # Set weights
    for i,l in enumerate(new_model.layers):
        l.set_weights(model.layers[i].get_weights())
    return new_model

def adapt_graph_to_new_input(model, input_shape, old_input_shape, verbose=False):
    old_config = model.get_config()
    config = old_config.copy()
    # Adapt input shape
    config['input_config'][0]['input_shape'] = input_shape
    # Get names of each layer
    keys = model.nodes.keys()
    # Get the flatten layer
    flatten_keys =[k for k in keys if  type(model.nodes[k]) is Flatten]
    # Get the name of the layer just before Flatten
    feature_key = keys[keys.index(flatten_keys[-1])-1]
    # Adapt pool size
    if config['nodes'][feature_key]["name"] == "Activation":
        # Merge case : multiple pooling size to adapt
        parents = get_parents(feature_key, config)
        for par in parents:
            # Store old pool size
            old_pool_size = config['nodes'][par]["pool_size"]
            # Compute Global stride
            global_stride = 1
            iteration = par
            count = 0
            while iteration!='input' and count < 100:
                count += 1
                iteration = get_parents(iteration, config)
                if iteration != 'input' and config['nodes'][iteration]["name"] == "MaxPooling2D":
                    global_stride *= config['nodes'][iteration]["pool_size"][0]
            # Compute new pool size
            new_pool_size = int(old_pool_size[0]+(input_shape[1]-old_input_shape[1])/global_stride)
            if verbose:
                print "Input shape :", input_shape
                print "Poolsize :", old_pool_size, "-> (%d,%d)"%(new_pool_size,new_pool_size)
            config['nodes'][par]["pool_size"] = (new_pool_size,new_pool_size)
            config['nodes'][par]["strides"] = (new_pool_size,new_pool_size)
    else:
        raise Exception("Not implemented")
    new_model = model_from_config(config)
    # Set weights
    for i,name in enumerate(new_model.nodes.keys()):
        new_model.nodes[name].set_weights(model.nodes[name].get_weights())
    return new_model

def get_parents(key, config):
    parents = [elem for elem in config['node_config'] if elem['name']==key]
    if parents[0]["input"] is not None:
        return parents[0]["input"]
    else:
        return parents[0]["inputs"]

def mean_with_list_axis(a, ax):
    out = np.mean(a, axis=ax[-1])
    if len(ax)>1:
        for i in ax[::-1][1:]:
            out = np.mean(out, axis=i)
    return out

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

def predict(model, dataset, training_params, flip, verbose=True):
    # New epoch
    dataset.on_new_epoch()
    N = training_params.Ntest/training_params.test_batch_size
    if N==0:
        raise Exception("Ntest = 0.")
    for i in range(N):
        if verbose:
            print "\rBatch %d over %d"%(i,N),
        # Get next batch
        batch,targets= dataset.get_batch()
        # Eventually flip
        if flip:
            batch = np.fliplr(batch.transpose(1,2,3,0)).transpose(3,0,1,2)
        # Preprocess
        for mode in training_params.valid_preprocessing:
            batch = preprocess_dataset(batch, training_params, mode)
        # Predict
        if type(model) is Graph:
            pred = model.predict({"input":np.array(batch.transpose(0,3,1,2), "float32")})["output"]
        else:
            pred = model.predict(np.array(batch.transpose(0,3,1,2), "float32"))
        # Accumulate preds
        if i==0:
            predictions = np.copy(pred)
            labels = np.copy(convert_labels(targets))
        else:
            predictions = np.concatenate((predictions,pred))
            labels = np.concatenate((labels,convert_labels(targets)))

    return predictions, labels

def multiscale_predict(model, training_params, division="leaderboard", verbose=False):
    initial_input_shape = model.input_shape
    k = 0
    for test_size in training_params.test_sizes:
        if verbose:
            print "\nTesting for size :" + str(test_size)
        # Get the best model
        if test_size[0] != model.input_shape[2] or test_size[1] != model.input_shape[3]:
            new_model = adapt_to_new_input(model, (test_size[2],test_size[0],test_size[1]), initial_input_shape[1:],
                                           verbose=True)
        else:
            new_model = model
        testset = FuelDataset("test", test_size, batch_size=training_params.test_batch_size, shuffle=False,
                              division=division)
        preds, labels  = predict(new_model, testset, training_params, flip=False, verbose=verbose)
        if k == 0:
            final_preds = np.copy(preds)
        else:
            final_preds += preds
        k+=1.0
        # Predictions on the flipped testset
        flipped_preds, labels = predict(new_model, testset, training_params, flip=True, verbose=verbose)
        final_preds += flipped_preds
        k+=1.0

    # Arithmetic averaging of predictions
    final_preds_arithm = final_preds/k

    return final_preds_arithm, labels

def test_model(model, dataset, training_params, flip=False, return_preds=False, verbose=True):
    """
    Expects a 4D dataset (N, rows, cols, channels).
    Expects binary labels.
    """
    # Prediction
    predictions, labels = predict(model, dataset, training_params, flip, verbose)
    if verbose:
        print "\r"
    # Compute metrics
    test_loss = categorical_crossentropy(labels, predictions).mean()
    count = np.sum(np.argmax(labels, axis=1) - np.argmax(predictions, axis=1) == 0)
    score = float(count)/labels.shape[0]
    if verbose:
        print "Accuracy = %.3f"%(score)
        print "Test Loss = %.3f"%(test_loss)
    if return_preds:
        return score, test_loss, predictions, labels
    else:
        return score, test_loss

def test_model_on_exp(training_params, verbose=False, write_txt_file=False):
    model, path_model = get_best_model_from_exp(training_params.path_out)
    initial_input_shape = model.input_shape
    print "\n" + path_model
    k = 0
    lines = []
    for test_size in training_params.test_sizes:
        if verbose:
            s = "\nTesting for size :" + str(test_size)
            print s
            lines.append(s)
        # Get the best model
        if test_size[0] != model.input_shape[2] or test_size[1] != model.input_shape[3]:
            new_model = adapt_to_new_input(model, (test_size[2],test_size[0],test_size[1]), initial_input_shape[1:],
                                           verbose=True)
        else:
            new_model = model

        testset = FuelDataset("test", test_size, batch_size=training_params.test_batch_size, shuffle=False,
                              division="not_leaderboard")
        score, loss, preds, labels  = test_model(new_model, testset, training_params,
                                                 flip=False, verbose=verbose, return_preds=True)
        if write_txt_file:
            lines.append("\n\tDraw testset score = %.5f\n\tDraw testset loss = %.5f"%(score,loss))
        if k == 0:
            final_preds = np.copy(preds)
        else:
            final_preds += preds
        k+=1.0
        # Predictions on the flipped testset
        flipped_score, flipped_loss, flipped_preds, labels = test_model(new_model, testset, training_params,
                                                                        flip=True, verbose=verbose, return_preds=True)
        if write_txt_file:
            lines.append("\n\tFlipped testset score = %.5f\n\tFlipped testset loss = %.5f"%(flipped_score,flipped_loss))
        final_preds += flipped_preds
        k+=1.0

    # Arithmetic averaging of predictions
    final_preds_arithm = final_preds/k
    count = np.sum(np.argmax(labels, axis=1) - np.argmax(final_preds_arithm, axis=1) == 0)
    final_score_arithm = float(count)/labels.shape[0]
    if verbose:
        s = "\nFinal score (arithm) =%.5f"%final_score_arithm
        print s
        lines.append(s)

    if write_txt_file:
        f = open(training_params.path_out+"/testset_score.txt", "w")
        for line in lines:
            f.writelines(line)
        f.close()

    return final_preds_arithm, final_score_arithm, labels

def test_ensemble_of_models(training_params, path_out=os.path.abspath("experiments/ensemble_of_models.txt"),
                            write_txt=True, verbose=True):
    predictions = []
    scores = []
    for i,path in enumerate(training_params.ensemble_models):
        # For each model, get the predictions
        training_params.path_out = path
        # Get predictions, No need to get the testset
        model_preds, model_score, labels = test_model_on_exp(training_params,
                                                             verbose=verbose, write_txt_file=False)
        training_params.test_sizes = [(270,270,3), (210,210,3)]
        # Accumulate predictions and scores
        predictions.append(model_preds)
        scores.append(model_score)
        if verbose:
            print "%s = %.5f"%(path, model_score)
    # Fusion
    final_predictions = np.mean(np.array(predictions), axis=0)
    count = np.sum(np.argmax(labels, axis=1) - np.argmax(final_predictions, axis=1) == 0)
    final_score = float(count)/labels.shape[0]
    if verbose:
        print "Ensemble Score = %.5f\n"%(final_score)
    # Write the result in a textfile
    if write_txt:
        f = open(path_out, "w")
        for i,path in enumerate(training_params.ensemble_models):
            f.writelines("%s = %.5f\n"%(path,scores[i]))
        f.writelines("Ensemble Score = %.5f\n"%(final_score))
        f.close()

def generate_submission_file(training_params, path_out=os.path.abspath("experiments/submission_file.txt"),
                            verbose=True):
    predictions = []
    for i,path in enumerate(training_params.ensemble_models):
        # For each model, get the predictions
        model, path_model = get_best_model_from_exp(path)
        if verbose:
            print "Model : %s"%(path_model)
        # Get predictions, No need to get the testset
        preds, labels = multiscale_predict(model, training_params, division="leaderboard",
                                                 verbose=verbose)
        # Accumulate predictions and scores
        predictions.append(preds)

    # Fusion
    final_predictions = np.mean(np.array(predictions), axis=0)
    binary_predictions = 1.0-np.argmax(final_predictions, axis=1) # Dogs->1, Cats->0
    # Write the result in a textfile
    f = open(path_out, "w")
    f.writelines("id,label\n")
    for i,label in enumerate(binary_predictions):
        f.writelines("%d,%d\n"%(i+1, int(label)))
    f.close()

def get_features(model, dataset, position, N, training_params, verbose, flip=False):

    intermediate_outputs = K.function([model.layers[0].input], [model.layers[position].get_output(train=False)])

    if N==0:
        raise Exception("Ntest = 0.")
    for i in range(N):
        if verbose:
            print "\rBatch %d over %d"%(i,N),
        # Get next batch
        batch,targets= dataset.get_batch()
        # Eventually flip
        if flip:
            batch = np.fliplr(batch.transpose(1,2,3,0)).transpose(3,0,1,2)
        # Preprocess
        for mode in training_params.valid_preprocessing:
            batch = preprocess_dataset(batch, training_params, mode)
        # Predict
        pred = intermediate_outputs([np.array(batch.transpose(0,3,1,2), "float32")])[0]
        # Accumulate preds
        if i==0:
            predictions = np.copy(pred)
            labels = np.copy(convert_labels(targets))
        else:
            predictions = np.concatenate((predictions,pred))
            labels = np.concatenate((labels,convert_labels(targets)))

    return predictions, labels

def get_features_on_exp(position, mode, N, training_params, verbose=False):

    model, path_model = get_best_model_from_exp(training_params.path_out)
    initial_input_shape = model.input_shape
    print "\n" + path_model
    k = 0
    out = []
    for test_size in training_params.test_sizes:
        if verbose:
            s = "\nTesting for size :" + str(test_size)
            print s
        # Get the best model
        if test_size[0] != model.input_shape[2] or test_size[1] != model.input_shape[3]:
            new_model = adapt_to_new_input(model, (test_size[2],test_size[0],test_size[1]), initial_input_shape[1:],
                                           verbose=True)
        else:
            new_model = model

        dataset = FuelDataset(mode, test_size, batch_size=training_params.test_batch_size, shuffle=False,
                              division=training_params.division)
        preds, labels  = get_features(new_model, dataset, position, N, training_params, True, flip=False)
        # Predictions on the flipped testset
        flipped_preds, flipped_labels  = get_features(new_model, dataset, position, N, training_params, True,
                                                      flip=False)
        out.append(preds)
        out.append(flipped_preds)
    return out, labels

def generate_csv_file(name, position, training_params):

    training_params.division = "not_leaderboard"
    subnames = ["150x150", "150x150_flipped","210x210", "210x210_flipped","270x270", "270x270_flipped"]

    mode = "test"
    N = training_params.Ntest/training_params.test_batch_size
    out, labels = get_features_on_exp(position, mode, N, training_params, True)
    # Write csv file
    path = training_params.path_out + "/" + name + "_" + mode + "_"
    for i,elem in enumerate(out):
        np.savetxt(path+subnames[i]+".csv", elem, delimiter=",")
    np.savetxt(path+"labels.csv", labels)

    mode = "valid"
    N = training_params.Nvalid/training_params.test_batch_size
    out, labels = get_features_on_exp(position, mode, N, training_params, True)
    # Write csv file
    path = training_params.path_out + "/" + name + "_" + mode + "_"
    for i,elem in enumerate(out):
        np.savetxt(path+subnames[i]+".csv", elem, delimiter=",")
    np.savetxt(path+"labels.csv", labels)

    mode = "train"
    N = training_params.Ntrain/training_params.test_batch_size
    out, labels = get_features_on_exp(position, mode, N, training_params, True)
    # Write csv file
    path = training_params.path_out + "/" + name + "_" + mode + "_"
    for i,elem in enumerate(out):
        np.savetxt(path+subnames[i]+".csv", elem, delimiter=",")
    np.savetxt(path+"labels.csv", labels)









