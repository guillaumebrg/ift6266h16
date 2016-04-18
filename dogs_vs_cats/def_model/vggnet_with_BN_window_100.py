__author__ = 'Guillaume'

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
import numpy as np


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
    model.add(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(MaxPooling2D(pool_size=(2,2)))
    ###
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(MaxPooling2D(pool_size=(2,2)))
    ###
    model.add(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(activation())
    model.add(MaxPooling2D(pool_size=(6,6)))

    # MLP
    model.add(Flatten())
    model.add(Dense(1024, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(activation())
    #model.add(Dropout(p=0.4))
    model.add(Dense(512, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(activation())
    #model.add(Dropout(p=0.4))
    model.add(Dense(2, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)

    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return model

