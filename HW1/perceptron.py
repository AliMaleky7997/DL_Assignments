import numpy as np


class Perceptron:
    def __init__(self, feature_dim, num_classes):
        """
        in this constructor you have to initialize the weights of the model with zeros. Do not forget to put 
        the bias term! 
        """
        pass
        
    def train(self, feature_vector, y):
        """
        this function gets a single training feature vector (feature_vector) with its label (y) and adjusts 
        the weights of the model with perceptron algorithm. 
        Hint: use self.predict() in your implementation.
        """
        pass

    def predict(self, feature_vector):
        """
        returns the predicted class (y-hat) for a single instance (feature vector).
        Hint: use np.argmax().
        """
        pass