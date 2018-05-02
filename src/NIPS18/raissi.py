"""
This script implements Maziar Raissi's algorithm
https://arxiv.org/abs/1804.07010


Note: with this method we cannot BatchNorm to speed up training, as we are using autograd.grad!!!!!!
"""

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce


cuda = False



def test_automatic_differentiation():
    """
    This function tests the automatic differentiation necessary for Raissi's algorithm
    """
    
    # test automatic differentiation
    x = Variable(torch.randn((2,10)), requires_grad=True)
    #layer = nn.Sequential(nn.Linear(10,1), nn.ReLU())#nn.BatchNorm1d(1), nn.ReLU())
    layer = nn.Linear(10,1)
    output = layer(x)
    
    l = []
    for row in range(x.size()[0]):
        l.append(torch.autograd.grad(output[row], x, create_graph=True)[0])
    output_grad = reduce(lambda x,y:x+y, l)
    output_grad
    
    list(layer.parameters())
    
    output_grad = torch.autograd.grad(output,x)
    
    # we have to make sure that the derivative of the output also depends on the model's parameters and therefore these will be taken into account when doing backpropagation
    # the model is f(x) = a+bx
    # The error is J(a,b) = (a+bx)^2 - df/dx = (a+bx)^2 - b
    # The partial derivatives are dJ/da = 2(a+bx), dJ/db = 2(a+bx)x - 1
    x = Variable(torch.randn((1,1)), requires_grad=True)
    layer = nn.Linear(1,1)
    layer.zero_grad()
    output = layer(x)
    l = []
    for row in range(x.size()[0]):
        l.append(torch.autograd.grad(output[row], x, create_graph=True)[0])
    output_grad = reduce(lambda x,y:x+y, l)
    output_grad
    
    error = output**2 - output_grad  # random error function depending on the output and the gradient of the output
    error.backward()
    for param in layer.parameters():
        print(param.grad)
    print('x = {}\n parameters = {}'.format(x, list(layer.parameters())))   # it works!!!!!!
    



class Net_Raissi(nn.Module):
    """
    Maziar Raissi's network
    https://arxiv.org/abs/1804.07010
    """
    
    def __init__(self, dim):
        super(Net_Raissi, self).__init__()
        self.dim = dim
        
        self.i_h1 =  self.hiddenLayer(dim, 256)
        self.h1_h2 = self.hiddenLayer(256, 256)
        self.h2_h3 = self.hiddenLayer(256, 256)
        self.h3_h4 = self.hiddenLayer(256, 256)
        self.h4_h5 = self.hiddenLayer(256, 256)
        self.h5_o = nn.Linear(256, 1)
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.Sigmoid())
        return layer
        
    def forward(self, x):
        # first coordinate of x is time
        h1 = self.i_h1(x)
        h2 = self.h1_h2(h1)
        h3 = self.h2_h3(h2)
        h4 = self.h3_h4(h3)
        h5 = self.h4_h5(h4)
        output = self.h5_o(h5)
        
        # automatic differentiation
        l = []
        for row in range(1,x.size()[0]):
            # first coordinate of x is time, so we start from 1
            l.append(torch.autograd.grad(output[row],x, create_graph=True)[0])
        output_grad = reduce(lambda x,y:x+y, l)
            
        return output, output_grad
    


model = Net_Raissi(10)
x = Variable(torch.randn((2,10)), requires_grad=True)
output, output_grad = model(x)


def train():
    
    batch_size = 100
    base_lr = 0.01
    dim = 10
    init_t = 0
    T = 1
    timestep = 0.01
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    sigma = 1
    
    model = Net_Raissi(dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
    
    n_iter = 100  # to be changed
    
    for it in range(n_iter):
        x = torch.zeros((batch_size, dim))
        t = torch.zeros(batch_size, 1)
        tx = Variable(torch.cat([t,x], dim=1))
        
        xi = Variable(torch.randn(x.data.size()))
        
        v, grad_v  = model(tx) 
        
        alpha = # to complete-math.sqrt(self.lambda_) * grad
        f = # to complete
        Y = []
        
        
        for i in range(1, timegrid.size):
            h = self.timegrid[i]-self.timegrid[i-1]
            xi = Variable(torch.randn(x.data.size()))
            
        
        
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    





