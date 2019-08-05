import numpy as np
from utils import Parameter, Layer, LossLayer
import math


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def relu(x):
    x[x < 0] = 0
    return x


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu_d(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class Relu(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Relu, self).__init__()
        self.cache = {}

    def forward(self, x, **kwargs):
        '''
        x: input to the forward pass which is a numpy 2darray
        kwargs: some extra inputs which are not used in Relu forward pass
        '''
        out = None
        ###########################################################################
        # TODO: Implement the relu forward pass.                                  #
        # Store the output in out and save whatever you need for backward pass in #
        # self.cache.                                                             #
        ###########################################################################
        self.cache['x'] = x
        out = x
        out[x < 0] = 0
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = None
        ###########################################################################
        # TODO: Implement the relu backward pass. Store gradient w.r.t. the input #
        # of the forward pass (i.e. x) in dx.                                     #
        ###########################################################################
        dx = dout * relu_d(self.cache['x'])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Relu'


class Sigmoid(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Sigmoid, self).__init__()
        self.cache = {}

    def forward(self, x, **kwargs):
        '''
        x: input to the forward pass which is a numpy 2darray
        kwargs: some extra inputs which are not used in Sigmoid forward pass
        '''
        out = None
        ###########################################################################
        # TODO: Implement the sigmoid forward pass. Store the result in out.      #
        # Store the variables required for backward pass in self.cache.           #
        ###########################################################################
        self.cache['x'] = x
        out = sigmoid(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = None
        ###########################################################################
        # TODO: Implement the sigmoid backward pass. Store gradient of loss w.r.t #
        # input of the forward pass (i.e. x) in dx.                               #
        ###########################################################################
        dx = dout * sigmoid_d(self.cache['x'])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Sigmoid'


class Tanh(Layer):
    def __init__(self):
        '''
        This layer is inherited from the class Layer in utils.py.
        '''
        super(Tanh, self).__init__()
        self.cache = {}

    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: some extra inputs which are not used in Tanh forward pass
        '''
        out = None
        ###########################################################################
        # TODO: Implement the tanh forward pass. Store the result in out. Store   #
        # the variables required for backward pass in self.cache.                 #
        # Hint: you can use np.tanh().                                            #
        ###########################################################################
        self.cache['x'] = x
        out = np.tanh(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        
        '''
        dx = None
        ###########################################################################
        # TODO: Implement the tanh backward pass. Store gradient of loss w.r.t    #
        # input of the forward pass (i.e. x) in dx.                               #
        ###########################################################################
        dx = dout * (1 - np.power(np.tanh(self.cache['x']), 2))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Tanh'


class SoftmaxCrossEntropy(LossLayer):
    def __init__(self):
        '''
        This layer is inherited from the LossLayer class in utils.py
        '''
        super(SoftmaxCrossEntropy, self).__init__()
        self.cache = {}

    def forward(self, x, y):
        '''
        x is the input to the layer which is a numpy 2d-array with shape (N, D)
        y contains the ground truth labels for instances in x which has the shape (N,)
        This function should do two things:
        1) Apply softmax activation on the input
        2) Compute the loss function using cross entropy loss function and returns the loss. 
        '''
        loss = None
        ###########################################################################
        # TODO: Implement the SoftmaxCrossEntropy forward pass. Store the         # 
        # computed loss in loss.                                                  #
        # Save whatever you need for backward pass in self.cache.                 #
        # You CANNOT use any for loops here and should implement only with numpy  #
        # vectorized operations.                                                  #
        # NOTE: Implement a numerically stable version of softmax. If you are not # 
        # careful here it is easy to run into numeric instability!                #
        ###########################################################################
        bias = ((np.min(x, axis=1) + np.max(x, axis=1)) / 2).reshape((-1, 1))
        x_nlize = x - bias
        e_x = np.exp(x_nlize)
        sigma = np.sum(e_x, axis=1).reshape((-1, 1))
        p = e_x / sigma
        y_oh = np.zeros_like(p)
        y_oh[np.arange(y_oh.shape[0]), y] = 1

        temp = y_oh * p
        temp = np.sum(temp, axis=1)
        temp = -np.log(temp)

        self.cache['y_hat'] = p
        self.cache['y'] = y_oh

        loss = np.mean(temp)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return loss

    def backward(self):
        dx = None
        ###########################################################################
        # TODO: Implement the SoftmaxCrossEntropy backward pass.                  #
        # You should compute the gradient of computed loss in the forward pass    #
        # w.r.t. the input x and store it in dx.                                  #
        # you CANNOT use any for loops in your implementation                     #
        ###########################################################################
        dx = (self.cache['y_hat'] - self.cache['y']) / self.cache['y'].shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Softmax Cross-Entropy'


class FullyConnected(Layer):
    def __init__(self, initial_value_w, initial_value_b, reg=0.):
        '''
        This layer is inherited from the class Layer in utils.py.
        initial_value_w: The inital value of weights
        initial_value_b: The initial value of biases
        reg: Regularization coefficient or strength used for L2-regularization
        Parameter class (in utils.py) is used for defining paramters
        '''
        super(FullyConnected, self).__init__()
        self.reg = reg
        self.params = {}
        self.params['w'] = Parameter('w', initial_value_w)
        self.params['b'] = Parameter('b', initial_value_b)

    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: some extra inputs which are not used in FullyConnected forward pass
        '''
        w, b = self.params['w'].data, self.params['b'].data
        out = None
        ###########################################################################
        # TODO: Implement the FullyConnected forward pass.                        #
        # Save the output in out.                                                 #
        # Save whatever you need for backward pass in self.cache.                 #
        ###########################################################################
        self.cache = {'x': x, 'w': w, 'b': b}
        out = np.matmul(x, w) + b
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        reg, w, b = self.reg, self.params['w'].data, self.params['b'].data
        dx, dw, db = None, None, None
        ###########################################################################
        # TODO: Implement the FullyConnected backward pass.                       #
        # Store the gradient of loss w.r.t. x, w, b in dx, dw, db repectively.    #
        # Don't forget to add gradient of L2-regularization term in loss w.r.t w  #
        # to dw!                                                                  #   
        ###########################################################################
        dx = np.matmul(dout, self.cache['w'].transpose())
        dw = np.matmul(self.cache['x'].transpose(), dout) + 2 * self.reg * w
        db = np.sum(dout, axis=0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        # storing the gradients in grad attribute of parameters
        self.params['w'].grad = dw
        self.params['b'].grad = db
        return dx

    def get_params(self):
        '''
        This function overrides the get_params method of class Layer.
        '''
        return self.params

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'FullyConnected {}'.format(self.params['w'].data.shape)

    def get_reg(self):
        return self.reg


class BatchNormalization(Layer):
    def __init__(self, gamma_initial_value, beta_initial_value, eps=1e-5, momentum=0.9):
        '''
        This layer is inherited from the Layer class in utils.py.
        '''
        super(BatchNormalization, self).__init__()
        self.params = {}
        self.eps = eps
        self.momentum = momentum
        self.running_mean = self.running_var = np.zeros_like(gamma_initial_value)

        self.params['gamma'] = Parameter('gamma', gamma_initial_value)
        self.params['beta'] = Parameter('beta', beta_initial_value)

    def forward(self, x, **kwargs):
        mode = kwargs.pop('mode')
        N, D = x.shape
        running_mean, running_var = self.running_mean, self.running_var
        momentum, gamma, beta = self.momentum, self.params['gamma'].data, self.params['beta'].data
        eps = self.eps
        out = None
        if mode == 'TRAIN':
            #######################################################################
            # TODO: Implement the training-time forward pass for batch norm.      #
            # Use minibatch statistics to compute the mean and variance, use      #
            # these statistics to normalize the incoming data, and scale and      #
            # shift the normalized data using gamma and beta.                     #
            #                                                                     #
            # You should store the output in the out. Store whatever you need for #                                                           # barckward pass in self.cache as a tuple.                            #
            #                                                                     #
            # You should also use your computed sample mean and variance together #
            # with the momentum variable to update the running mean and running   #
            # variance, storing your result in the running_mean and running_var   #
            # variables.                                                          #
            #                                                                     #
            # Note that though you should be keeping track of the running         #
            # variance, you should normalize the data based on the standard       #
            # deviation (square root of variance) instead!                        # 
            # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
            # might prove to be helpful.                                          #
            #######################################################################
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            u = (x - mean) / np.sqrt(var + eps)
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
            out = gamma * u + beta
            self.cache = {'u': u, 'x': x, 'mean': mean, 'var': var}

            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        elif mode == 'TEST':
            #######################################################################
            # TODO: Implement the test-time forward pass for batch normalization. #
            # Use the running mean and variance to normalize the incoming data,   #
            # then scale and shift the normalized data using gamma and beta.      #
            # Store the result in the out variable.                               #
            #######################################################################
            u = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * u + beta
            #######################################################################
            #                          END OF YOUR CODE                           #
            #######################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        self.running_mean = running_mean
        self.running_var = running_var

        return out

    def backward(self, dout):
        gamma, beta = self.params['gamma'].data, self.params['beta'].data
        dx, dgamma, dbeta = None, None, None
        ###########################################################################
        # TODO: Implement the backward pass for batch normalization. Store the    #
        # results in the dx, dgamma, and dbeta variables.                         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
        # might prove to be helpful.                                              #
        ###########################################################################
        x = self.cache['x']
        u = self.cache['u']
        N, D = x.shape
        mean = self.cache['mean']
        x0 = self.cache['x'] - mean
        var = self.cache['var'] + self.eps
        std = np.sqrt(var)
        du = dout * gamma

        var_dx = -np.sum(du * x0, axis=0)/(2*var*std)
        mean_dx = np.sum(-du/std, axis=0) - 2 * var_dx * np.mean(x0, axis=0)
        dx = (2*var_dx*x0+mean_dx)/N + du/std

        dgamma = np.sum(dout*u, axis=0)
        dbeta = np.sum(dout, axis=0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        # saving the gradients in grad attribute of parameters
        self.params['gamma'].grad = dgamma
        self.params['beta'].grad = dbeta
        return dx

    def get_params(self):
        return self.params

    def reset(self):
        self.running_var = self.running_mean = np.zeros_like(self.running_mean)

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Batch Normalization eps={}, momentum={}'.format(self.eps, self.momentum)


class Dropout(Layer):
    def __init__(self, p):
        '''
        This layer is inherited from the class Layer in utils.py.
        p: probability of keeping a neuron active.
        '''
        super(Dropout, self).__init__()
        self.p = p
        self.cache = {}

    def forward(self, x, **kwargs):
        '''
        x: A numpy 2darray which is the input to the forward pass
        kwargs: Some extra input from which mode is used for dropout forward pass
        '''
        mask, out, p = None, None, self.p
        mode = kwargs.pop('mode')
        if mode == 'TRAIN':
            #######################################################################
            # TODO: Implement training phase forward pass for inverted dropout    #
            # and save the output in out.                                         #
            # Store the dropout mask in the mask variable and store it in         #
            # self.cache to be used in backward pass.                             #
            #######################################################################
            mask = np.random.binomial(1, p, size=x.shape)/p
            out = x * mask

            self.cache['mask'] = mask*p
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        elif mode == 'TEST':
            #######################################################################
            # TODO: Implement the test phase forward pass for inverted dropout.   #
            #######################################################################
            out = x
            #######################################################################
            #                            END OF YOUR CODE                         #
            #######################################################################
        else:
            raise ValueError('Invalide mode')

        out = out.astype(x.dtype, copy=False)
        return out

    def backward(self, dout):
        '''
        dout: upstream gradient of loss w.r.t. the output of forward pass
        '''
        dx = None
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * self.cache['mask']
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return dx

    def __str__(self):
        '''
        when you call print function on an instance of the class this function is called.
        '''
        return 'Dropout p={}'.format(self.p)
