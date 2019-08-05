import numpy as np


def sgd(param, dparam, config=None):
    """
    Performs stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    param -= config['learning_rate'] * dparam
    return param, config


def sgd_momentum(param, dparam, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as param and dparam used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(param))
    next_param = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_param variable. You should also use momentum and learning_rate #
    # to update the velocity v.                                               # 
    ###########################################################################
    v = config['momentum'] * v - config['learning_rate'] * dparam
    param += v
    next_param = param
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v
    return next_param, config


def rmsprop(param, dparam, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(param))
    
    next_param = None
    ###############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of param #
    # in the next_param variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                            #
    ###############################################################################
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dparam ** 2
    param += - config['learning_rate'] * dparam / (np.sqrt(config['cache']) + config['epsilon'])
    next_param = param
    ###############################################################################
    #                             END OF YOUR CODE                                #
    ###############################################################################

    return next_param, config
    


def adam(param, dparam, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(param))
    config.setdefault('v', np.zeros_like(param))
    config.setdefault('t', 0)

    next_param = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t BEFORE    #
    # using it in any calculations.                                           #
    ###########################################################################
    config['t'] += 1

    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dparam
    mt = config['m'] / (1 - config['beta1'] ** config['t'])

    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dparam ** 2
    vt = config['v'] / (1 - config['beta2'] ** config['t'])

    param += - (config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon']))
    next_param = param
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return next_param, config
