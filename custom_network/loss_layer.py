import sys
import numpy as np
from layer import Layer

class LossLayer(Layer):

    def __init__(self, ip_dim, n_neurons, learning_rate, activation_fn='softmax',
    gradient_method='default', layer_id=None, loss_func='cross_entropy') -> None:
        super(LossLayer, self).__init__(ip_dim, n_neurons, learning_rate, activation_fn, 
        gradient_method, layer_id=layer_id)
        self.loss_func = loss_func
        self.loss = None
        self.dE_dz = None
        self.dE_dy = None

    def compute_loss(self, y, labels):

        N = labels.shape[1]
        self.dE_dz = None
        
        # for cross entropy loss dE/dz is returned
        if self.loss_func == 'cross_entropy' and self.activation_fn == 'softmax':
            eps = 1e-6
            y = np.where(y + eps > 1.0, 1.0, y+eps)
            ce_error = -labels * np.log(y)
            self.loss = (1./N) * np.sum(ce_error)
            self.dE_dz = y - labels
        
        elif self.loss_func == 'mse' and self.activation_fn in ['linear', 'relu', 'sigmoid']:
            self.loss = np.mean(0.5*np.square(y - labels))
            self.dE_dy = y - labels

        return self.loss
    
    def backward(self, ignore):

        if self.activation_fn != 'softmax':
            return super().backward(self.dE_dy)

        # compute the gradient:
        # change in the network output wrt the change 
        # in the weights of the current layer
        # dE_dw = dz_dw * dy_dz * dE_dy
        # dE_dw = dz_dw * dE_dz
        dE_dz = self.dE_dz
        dz_dw = self.x
        N = self.x.shape[-1]
        self.grad = (1./N) * np.dot(dE_dz, np.transpose(dz_dw))

        assert self.grad.shape == self.weights.shape

        self.dE_dx = self.compute_dE_dx(dE_dz)

        return self.dE_dx
    