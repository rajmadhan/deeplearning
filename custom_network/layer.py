import sys
import numpy as np
from scipy.special import softmax
from gradient import Gradient


class Layer:

    def __init__(self, ip_dim, n_neurons, learning_rate, activation_fn='relu',
    gradient_method='default', layer_id=None) -> None:
        
        self.ip_dim = ip_dim
        self.n_neurons = n_neurons
        self.activation_fn = activation_fn
        # perform weight initialization based on the activation function
        # reference: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        if activation_fn == 'relu':
            self.weights = np.math.sqrt(2/self.ip_dim) * np.random.randn(self.n_neurons, self.ip_dim + 1)
        else:
            lower, upper = -1.0/np.math.sqrt(self.ip_dim), 1.0/np.math.sqrt(self.ip_dim)
            self.weights = lower + np.random.rand(self.n_neurons, self.ip_dim + 1) * (upper - lower)
        self.grad = np.zeros(self.weights.shape)
        self.z = None
        self.y = None
        self.x = None
        self.dE_dx = None
        self.dy_dz = None        
        self.gradient_fn = Gradient(learning_rate, gradient_method)
        self.layer_id = layer_id
    
    def forward(self, X):

        self.x = np.vstack((X, np.ones((1, X.shape[-1]), dtype=np.float32)))
        self.z = np.dot(self.weights, self.x)

        if self.activation_fn == 'relu':
            self.y = np.where(self.z > 0, self.z, 0)

        elif self.activation_fn == 'sigmoid':
            self.y = 1./(1 + np.math.exp(-self.z))
        
        elif self.activation_fn == 'softmax':
            self.y = softmax(self.z, axis=0)

        return self.y

    def backward(self, dE_dy):

        # compute the gradient:
        # change in the network output wrt the change 
        # in the weights of the current layer
        # dE_dw = dz_dw * dy_dz * dE_dy
        # dE_dw = dz_dw * dE_dz
        if self.activation_fn == 'relu':
            dy_dz = 1 #np.ones(self.y.shape, dtype=np.float32)
        elif self.activation_fn == 'sigmod':
            dy_dz = self.y * (1 - self.y)
        dE_dz = dy_dz * dE_dy
        # softmax is only supported at the LossLayer
        # elif self.activation_fn == 'softmax':
        #     dE_dz = dE_dy
        dz_dw = self.x
        N = self.x.shape[-1]
        self.grad = (1./N) * np.dot(dE_dz, np.transpose(dz_dw))

        assert self.grad.shape == self.weights.shape

        self.dE_dx = self.compute_dE_dx(dE_dz)

        return self.dE_dx

    def compute_dE_dx(self, dE_dz):
        # compute the change in output wrt the
        # input of the current layer
        # dE_dx = SUM_j (dz_dx * dy_dz * dE_dy)
        # dE_dx = SUM_j (dz_dx * dE_dz)
        if self.layer_id != 1:
            dz_dx = self.weights
            # dy_dx = dz_dx * dy_dz
            # self.dE_dx = np.dot(dy_dx, dE_dx)
            dE_dx = np.dot(np.transpose(dz_dx), dE_dz)
            
            return dE_dx[:-1]        

    def apply_gradient(self):
        
        self.weights = self.gradient_fn.compute(self.weights, self.grad)
