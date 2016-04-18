__author__ = 'Guillaume'

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np
import pickle


def define_model(lr, momentum):
    # CONFIG
    model = Sequential()

    def activation():
        #return Activation("relu")
        return PReLU(alpha=0.25)

    # Create Layers
    # CONVNET
    model.add(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    ###
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(MaxPooling2D(pool_size=(2,2)))
    ###
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    ### RECURRENT PART
    model.add(Reshape((128, 25*25)))
    model.add(Permute((2,1)))
    model.add(LSTM(64, input_length=25*25, return_sequences=True))
    model.add(LSTM(64, input_length=25*25))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    pretrained = True
    trainable = False
    if pretrained:
        weight_path = "pretrained_models/win46_conv.pkl"
        with open(weight_path, "r") as f:
            weights = pickle.load(f)
        for i,w in enumerate(weights):
            model.layers[i].set_weights(w)
            model.layers[i].trainable = trainable

    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)

    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return model

