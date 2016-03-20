__author__ = 'Guillaume'

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Flatten, Dropout, Activation, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np


def define_model(lr, momentum):
    # CONFIG
    # CONFIG
    model_window_46 = Sequential()
    # Create Layers
    # CONVNET
    layers = []
    layers.append(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(25,25)))

    for layer in layers:
        model_window_46.add(layer)

    model_window_100 = Sequential()
    # Create Layers
    # CONVNET
    layers = []
    layers.append(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(6,6)))

    for layer in layers:
        model_window_100.add(layer)

    weights = np.load("dogs_vs_cats/pretrained_models/merge_w46_w100_weights.npy")

    count = 0
    for lay in model_window_46.layers:
        lay.set_weights(weights[count])
        #lay.trainable=False
        count += 1
    # Model w100
    for lay in model_window_100.layers:
        lay.set_weights(weights[count])
        #lay.trainable=False
        count += 1

    final_model = Sequential()
    final_model.add(Merge([model_window_46, model_window_100], mode="concat", concat_axis=1))

    layers = []
    # MLP
    layers.append(Flatten(input_shape=(384,1,1)))
    layers.append(Dense(1024, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("relu"))
    #layers.append(Dropout(p=0.4))
    layers.append(Dense(512, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("relu"))
    layers.append(Dropout(p=0.5))
    layers.append(Dense(2, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("softmax"))

    # Adding Layers
    for layer in layers:
        final_model.add(layer)

    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)

    final_model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return final_model


def define_graph(lr, momentum):
    # CONFIG
    # CONFIG
    model = Graph()
    model.add_input(name="input", input_shape=(3,150,150))
    # Create Layers
    # CONVNET
    layers = []
    layers.append(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(25,25)))

    model.add_node(layers[0], name="win46_layer0", input="input")
    for i,layer in enumerate(layers[1:]):
        model.add_node(layer, name="win46_layer%d"%(i+1), input="win46_layer%d"%(i))
    last_win46 = "win46_layer%d"%(i+1)

    # Create Layers
    # CONVNET
    layers = []
    layers.append(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(Activation("relu"))
    layers.append(MaxPooling2D(pool_size=(6,6)))

    model.add_node(layers[0], name="win100_layer0", input="input")
    for i,layer in enumerate(layers[1:]):
        model.add_node(layer, name="win100_layer%d"%(i+1), input="win100_layer%d"%(i))
    last_win100 = "win100_layer%d"%(i+1)

    #weights = np.load("dogs_vs_cats/pretrained_models/merge_w46_w100_weights.npy")

    model.add_node(Activation("linear"), "features_merge", inputs=[last_win46, last_win100],
                   merge_mode='concat', concat_axis=1)

    layers = []
    # MLP
    layers.append(Flatten(input_shape=(384,1,1)))
    layers.append(Dense(1024, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("relu"))
    #layers.append(Dropout(p=0.4))
    layers.append(Dense(512, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("relu"))
    layers.append(Dropout(p=0.5))
    layers.append(Dense(2, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("softmax"))

    # Adding Layers
    model.add_node(layers[0], name="fc_layer0", input="features_merge")
    for i,layer in enumerate(layers[1:]):
        model.add_node(layer, name="fc_layer%d"%(i+1), input="fc_layer%d"%(i))

    model.add_output(name="output", input="fc_layer%d"%(i+1))
    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)

    model.compile(loss={"output":"categorical_crossentropy"}, optimizer=sgd)

    return model
