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

    def compute_loss(self, y, labels):

        N = labels.shape[0]
        self.dE_dz = None
        
        assert self.loss_func == 'cross_entropy', print('loss function not supported')
        assert self.activation_fn == 'softmax', print('activation function not supported')

        # for cross entropy loss dE/dz is returned
        if self.loss_func == 'cross_entropy' and self.activation_fn == 'softmax':
            eps = 1e-6
            y = np.where(y + eps > 1.0, 1.0, y+eps)
            ce_error = -labels * np.log(y)
            self.loss = (1./N) * np.sum(ce_error)
            self.dE_dz = y - labels

        return self.loss
    
    def backward(self, ignore):

        # compute the gradient:
        # change in the network output wrt the change 
        # in the weights of the current layer
        # dE_dw = dz_dw * dy_dz * dE_dy
        # dE_dw = dz_dw * dE_dz
        dE_dz = self.dE_dz
        dz_dw = self.x
        N = self.x.shape[0]
        self.grad = (1./N) * np.dot(np.transpose(dz_dw), dE_dz)

        assert self.grad.shape == self.weights.shape

        self.dE_dx = self.compute_dE_dx(dE_dz)

        return self.dE_dx
    