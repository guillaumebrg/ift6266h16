__author__ = 'Guillaume'

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def define_model(lr, momentum):
    # CONFIG
    model = Sequential()

    # Create Layers
    # CONVNET
    layers = []
    layers.append(Convolution2D(8, 3, 3, activation = "relu", input_shape=(1,150,150)))
    layers.append(Convolution2D(8, 3, 3, activation = "relu"))
    layers.append(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    #layers.append(MaxPooling2D(pool_size=(2,2))) #if image is 100x100
    layers.append(Convolution2D(16, 3, 3, activation = "relu"))
    layers.append(Convolution2D(16, 3, 3, activation = "relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(32, 3, 3, activation = "relu"))
    layers.append(Convolution2D(32, 3, 3, activation = "relu"))
    layers.append(Convolution2D(32, 3, 3, activation = "relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))
    layers.append(Convolution2D(64, 3, 3, activation = "relu"))
    layers.append(Convolution2D(64, 3, 3, activation = "relu"))
    layers.append(Convolution2D(64, 3, 3, activation = "relu"))
    layers.append(MaxPooling2D(pool_size=(2,2)))

    # MLP
    layers.append(Flatten())
    layers.append(Dense(1024, activation="relu"))
    layers.append(Dense(512, activation="relu"))
    layers.append(Dense(2, activation="softmax"))

    # Adding Layers
    for layer in layers:
        model.add(layer)

    # COMPILE (learning rate, momentum, objective...)
    sgd = SGD(lr=lr, momentum=momentum)

    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    return model

