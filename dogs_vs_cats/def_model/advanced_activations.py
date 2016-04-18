__author__ = 'Guillaume'

from keras import initializations
from keras.layers.core import MaskedLayer
from keras import backend as K
import numpy as np

class PReLU(MaskedLayer):
    '''
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments:
        init: initialization function for the weights.
        weights: initial weights, as a list of a single numpy array.

    # References:
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='zero', alpha=None, weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.initial_alpha = alpha
        self.axis = 1
        super(PReLU, self).__init__(**kwargs)

    def build(self):
        shape = (self.input_shape[self.axis],)
        self.alphas = self.init(shape,
                                name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_alpha is not None:
            self.initial_weights = [np.array([self.initial_alpha for i in range(shape[0])])]
            del self.initial_alpha

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        input_shape = self.input_shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        X = self.get_input(train)
        pos = K.relu(X)
        a = K.reshape(self.alphas, broadcast_shape)
        neg = a * (X - abs(X)) * 0.5
        return pos + neg

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'init': self.init.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
