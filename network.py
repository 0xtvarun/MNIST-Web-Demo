import numpy as np

class Network(object):

    def __init__(self, weights, biases):
        self.biases = biases
        self.weights = weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    
    hyper_params = np.load('./data/model.npz')
    net = Network(hyper_param['weights'], hyper_param['biases'])
    
