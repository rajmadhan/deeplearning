import sys
import numpy as np
from layer import Layer
from loss_layer import LossLayer


class Network:

    def __init__(self, layer_size, activation_fn, learning_rate=0.001, gradient_method='default',
    loss_layer_activation_fn='softmax', loss_func='cross_entropy') -> None:
        
        self.layer_size = layer_size
        self.activation_fn = activation_fn
        self.lr = learning_rate
        self.gradient_method = gradient_method
        self.loss_layer_activation_fn = loss_layer_activation_fn
        self.loss_func = loss_func
        self.layers = []
        self.loss_layer = None

        # create network
        self.create_network()
    
    def create_network(self):

        for i in range(len(self.layer_size)-2):

            ip_dim = self.layer_size[i]
            n_neurons = self.layer_size[i+1]
            layer = Layer(ip_dim=ip_dim, n_neurons=n_neurons, learning_rate=self.lr,
            activation_fn=self.activation_fn, gradient_method=self.gradient_method,
            layer_id=i+1)
            self.layers.append(layer)

        ip_dim = self.layer_size[-2]
        n_neurons = self.layer_size[-1]
        self.loss_layer = LossLayer(ip_dim=ip_dim, n_neurons=n_neurons, learning_rate=self.lr,
            activation_fn=self.loss_layer_activation_fn, gradient_method=self.gradient_method,
            layer_id=len(self.layer_size)-1, loss_func=self.loss_func)
        self.layers.append(self.loss_layer)

        # self.loss_layer = LossLayer(loss_func=self.loss_func)
        # self.loss_fn = loss_layer.compute_loss
    
    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)        
        return X

    def compute_loss(self, pred, gt):

        total_loss = self.loss_layer.compute_loss(y=pred, labels=gt)
        return total_loss

    def backward(self):

        dE_dy = None
        for layer in self.layers[::-1]:
            dE_dy = layer.backward(dE_dy)

    def apply_gradient(self):

        for layer in self.layers:
            layer.apply_gradient()
            