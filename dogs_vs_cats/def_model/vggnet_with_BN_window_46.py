__author__ = 'Guillaume'

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2, ActivityRegularizer
import numpy as np

def define_model(lr, momentum):
    # CONFIG
    model = Sequential()

    def activation():
        #return Activation("relu")
        return PReLU(alpha=0.25)

    # Create Layers
    # CONVNET
    layers = []
    layers.append(ZeroPadding2D((1, 1), input_shape=(3,150,150)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(ZeroPadding2D((1, 1)))
    layers.append(Convolution2D(128, 3, 3, activation = "linear"))#, activity_regularizer=ActivityRegularizer(l1=0.01)))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization(axis=1))
    layers.append(activation())
    layers.append(MaxPooling2D(pool_size=(25,25)))

    # MLP
    layers.append(Flatten())
    #layers.append(Dropout(p=0.5))
    layers.append(Dense(512, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(activation())
    #layers.append(Dropout(p=0.4))
    layers.append(Dense(256, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(activation())
    #layers.append(Dropout(p=0.4))
    layers.append(Dense(2, activation="linear"))#, W_regularizer=l2(0.0001)))
    layers.append(BatchNormalization())
    layers.append(Activation("softmax"))

    # Adding Layers
    for layer in layers:
        model.add(layer)

    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)

    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return model

