__author__ = 'Guillaume'

import numpy as np
import pickle
import time

class Layer(object):  # inspired from BART VAN MERRIENBOER's notebook
    # __init__ is the constructor method of this class
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.randn(input_dim, output_dim)  # weights
        self.B = np.zeros(output_dim)  # bias
        self.activation = activation # activation
        self.input_ = None
        self.tmp = None # features before activation
        self.output_ = None

    def forward_propagation(self, input_):
        self.input_ = np.copy(input_)
        self.tmp = self.compute_features(input_)
        self.output_ = self.activation.activate(self.tmp)
        return self.output_

    def compute_features(self, input_):
        # specified in each subclass of Layer
        # examples :
        #   - Neurons -> affine transformation
        #   - Convolutional Layer -> convolution...
        pass

    def backward_propagation(self, gradient_wrt_output):
        # specified in each subclass of Layer
        pass

    def update_parameters(self, gradient_wrt_output, learning_rate):
        # specified in each subclass of Layer
        pass

class Activation(object):
    # By default it is Linear activation
    def activate(self, input_):
        return input_

    def gradient(self, input_, output_, previous_grad):
        return previous_grad

class Networks(object):
    def __init__(self, layers):
        # A network is composed by a list of layers
        self.layers = layers

    def forward_pass(self, input_):
        if input_.ndim == 1:
            input_ = input_[None,:]
        output = np.copy(input_)
        # Propagate through each layer : bottom to top
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def backward_pass(self, gradient_cost, learning_rate):
        gradients = [gradient_cost]
        # Back-Propagate through each layer : top to bottom
        for layer in self.layers[::-1]:
            # Update weights
            layer.update_parameters(gradients[-1], learning_rate)
            # Back propagate the gradient
            gradients.append(layer.backward_propagation(gradients[-1]))

    def return_gradients(self, gradient_cost):
        # function that returns analytical dJ/dW and dJ/dB for each layer
        # weights are not updated
        # can be used to compare analytical gradient with numerical gradients
        gradients = [gradient_cost]
        dW_and_dB = []
        for layer in self.layers[::-1]:
            dW_and_dB.append(layer.return_updates(gradients[-1]))
            gradients.append(layer.backward_propagation(gradients[-1]))
        return dW_and_dB

    def gradients_by_finite_differences(self, x, y, cost_function, eps=0.0001):
        # Function that returns numerical dJ/dW and dJ/dB for each layer
        # using finite differences method : df/dx = f(x+eps)-f(x) / eps
        # Weights are not updated
        # Can be used to compare analytical gradient with numerical gradients
        dW_and_dB = []
        # Compute the output before changing parameters
        out = self.forward_pass(x)
        Jout = cost_function.compute_cost(out, y)
        # Computing each partial derivative one by one
        for layer in self.layers[::-1]:
            dW = np.zeros(layer.W.shape)
            dB = np.zeros(layer.B.shape)
            # Fill dW
            for i in range(dW.shape[0]):
                for j in range(dW.shape[1]):
                    # Add eps to W[i,j]
                    layer.W[i,j] += eps
                    # Compute the new cost
                    Jout_eps = cost_function.compute_cost(self.forward_pass(x),y)
                    # Compute the partial derivative
                    dW[i,j] = (Jout_eps - Jout)/eps
                    # Remove eps
                    layer.W[i,j] -= eps
            # Fill dB
            for i in range(dB.shape[0]):
                # Add eps to W[i,j]
                layer.B[i] += eps
                # Compute the new cost
                Jout_eps = cost_function.compute_cost(self.forward_pass(x),y)
                # Compute the partial derivative
                dB[i] = (Jout_eps - Jout)/eps
                # Remove eps
                layer.B[i] -= eps
            dW_and_dB.append((dW,dB))
        return dW_and_dB

    def check_gradient(self, x, y, cost_function, rtol=10**-4, atol=10**-4, eps=0.0001, display=True):
        # Functions that check if numerical and analytical are equals (or close enough)
        out = self.forward_pass(x)
        dcost = cost_function.gradient(out, y)
        # Analytical gradients
        analytical_gradient = self.return_gradients(dcost)
        if display:
            print "### Analytical Gradients ###"
            for count,grad in enumerate(analytical_gradient[::-1]):
                print "Layer", count+1
                print "dW"
                print grad[0]
                print "dB"
                print grad[1]
                print
        # Numerical gradients
        finite_gradients = self.gradients_by_finite_differences(x, y, cost_function, eps=0.0001)
        if display:
            print "### Numerical Gradients ###"
            for count,grad in enumerate(finite_gradients[::-1]):
                print "Layer", count+1
                print "dW"
                print grad[0]
                print "dB"
                print grad[1]
                print
        # Check if gradients are equals : if not, it will raise an Assertion Error
        for grad1,grad2 in zip(analytical_gradient,finite_gradients):
            # Checking dW
            np.testing.assert_allclose(grad1[0],grad2[0], rtol=rtol, atol=atol)
            # Checking dB
            np.testing.assert_allclose(grad1[1],grad2[1], rtol=rtol, atol=atol)

        print "Gradients are OK."
        return True

    def save_weights(self, filename):
        # Function that permits to save weights as a pickle file
        f = open(filename, "wb")
        for layer in self.layers:
            pickle.dump(layer.W, f)
            pickle.dump(layer.B, f)
        f.close()

    def load_weights(self, filename):
        # Function that permits to extract weights from a pickle file.
        # If weights were saved using the 'save_weights' function.
        f = open(filename, "rb")
        try:
            for layer in self.layers:
                layer.W = pickle.load(f)
                layer.B = pickle.load(f)
        except:
            "Architecture and stored weights don't match."

    def train_model(self, trainset,  validset, cost_function, learning_rate=0.01, nepochs_max=1, max_no_best=1, batch_size=1, nexample_per_epoch=None, save_name="best_model.pkl"):
        # Dataset have to be given as tuples : (inputs[nexamples,:],targets[nexamples,:])

        train_cost_history = []
        valid_cost_history = []
        time_history = [0]

        # Training loop
        epoch = 0
        nbest = 0
        display = ""
        N = trainset[0].shape[0]
        if nexample_per_epoch is None: # if not specified, train over all available examples
            nexample_per_epoch = N

        # Compute cost before training
        # Compute Trainset cost
        out = self.forward_pass(trainset[0])
        train_cost = cost_function.compute_cost(out, trainset[1])
        train_cost_history.append(train_cost)
        # Compute Validation cost
        out = self.forward_pass(validset[0])
        valid_cost = cost_function.compute_cost(out, validset[1])
        best = valid_cost
        valid_cost_history.append(valid_cost)
        # Display
        print "\rInit   \t Train Cost = %.2f \t Valid Cost = %.2f \t Best Cost = %.2f"%(train_cost,valid_cost,best)

        while epoch < nepochs_max and nbest < max_no_best:
            start = time.time()
            # Create a random index
            index = np.arange(N)
            np.random.shuffle(index)
            # Loop over each example
            epoch +=1
            for i in np.arange(0,nexample_per_epoch,batch_size):
                # Forward pass
                out = self.forward_pass(trainset[0][index[i:(i+batch_size)]])
                # Gradient of the cost
                gradient_cost = cost_function.gradient(out, trainset[1][index[i:(i+batch_size)]])
                # Backward pass
                self.backward_pass(gradient_cost, learning_rate)
                print "\r New epoch : %d percent..."%(100*i/nexample_per_epoch),
            end = time.time()
            # only the time spent to update the weights is taken into account
            # time spent on cost evaluations, and model saving, is not recorded
            time_history.append(int(end-start))

            # Compute Trainset cost
            out = self.forward_pass(trainset[0])
            train_cost = cost_function.compute_cost(out, trainset[1])
            train_cost_history.append(train_cost)

            # Compute Validation cost
            out = self.forward_pass(validset[0])
            valid_cost = cost_function.compute_cost(out, validset[1])
            valid_cost_history.append(valid_cost)

            # Best score ?
            if valid_cost < best:
                best = valid_cost
                self.save_weights(save_name)
                nbest = 0
            else:
                nbest += 1

            print "\rEpoch %d \t Train Cost = %.2f \t Valid Cost = %.2f \t Best Cost = %.2f \t Time = %ds"%(epoch,train_cost,valid_cost,best, int(end-start))

        return valid_cost_history, train_cost_history, time_history

############# SUB CLASSES ###############

# NETWORKS subclasses
class MLP(Networks):
    def add(self, input_dim, output_dim, activation):
        # permits to add Neurons layers
        if self.layers: # if not empty
            if self.layers[-1].W.shape[1]!=input_dim:
                print "Inconsistent dimensions"
                pass
        self.layers.append(Neurons(input_dim, output_dim, activation))


# LAYER subclasses
class Neurons(Layer):
    def compute_features(self, input_):
        return input_.dot(self.W)+self.B

    def backward_propagation(self, gradient_wrt_output):
        return (self.activation.gradient(self.tmp, self.output_, gradient_wrt_output)).dot(self.W.T)

    def update_parameters(self, gradient_wrt_output, learning_rate):
        self.W -= learning_rate * self.input_.T.dot(self.activation.gradient(self.tmp, self.output_, gradient_wrt_output))
        self.B -= learning_rate * (self.activation.gradient(self.tmp, self.output_, gradient_wrt_output)).sum(axis=0)

    def return_updates(self, gradient_wrt_output):
        dW = self.input_.T.dot(self.activation.gradient(self.tmp, self.output_, gradient_wrt_output))
        dB = (self.activation.gradient(self.tmp, self.output_, gradient_wrt_output)).sum(axis=0)
        return dW,dB

# ACTIVATION subclasses
class Linear(Activation):
    def activate(self, input_):
        return input_

    def gradient(self, input_, output_, previous_grad):
        return previous_grad

class Sigmoid(Activation):
    def activate(self, input_):
        return 1.0/(1.0 + np.exp(-input_))

    def gradient(self, input_, output_, previous_grad):
        return previous_grad*(output_-output_**2)

class Softmax(Activation):
    def activate(self, input_):
        tmp = np.exp(input_)
        return tmp/tmp.sum(axis=1)[:,None]

    def gradient(self, input_, output_, previous_grad):
        # Compute the Jacobian matrix
        # In fact, I compute the jacobian matrix as 3D tensor to handle mini batch scenario
        jacob = np.repeat(output_[:,:,None], output_.shape[1], axis=2)
        jacob *= (np.repeat(np.eye(output_.shape[1])[None,:,:] , output_.shape[0], axis=0) -np.repeat(output_[:,None,:], output_.shape[1], axis=1))
        # Return the gradient
        return np.dot(previous_grad, jacob.T)[range(output_.shape[0]),:,range(output_.shape[0])]









