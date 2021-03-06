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
import math
from collections import namedtuple
import time


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
    # the example model we use is f(x) = a+bx
    # The error we use is J(a,b) = (a+bx)^2 - df/dx = (a+bx)^2 - b
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
        
        self.i_h1 =  self.hiddenLayer(dim+1, 256)   # dim+1 because the input to the network is (t,x)
        self.h1_h2 = self.hiddenLayer(256, 256)
        self.h2_h3 = self.hiddenLayer(256, 256)
        self.h3_h4 = self.hiddenLayer(256, 256)
        self.h4_h5 = self.hiddenLayer(256, 256)
        self.h5_o = nn.Linear(256, 1)
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.ReLU())
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
        #l = []
        #for row in range(x.size()[0]):
        #    l.append(torch.autograd.grad(output[row],x, create_graph=True)[0])
        #    #l.append(torch.autograd.grad(output[row],x)[0])#, create_graph=True)[0])
        #output_grad = reduce(lambda x,y:x+y, l)
        l = [output[row] for row in range(output.size()[0])]
        output_grad = torch.autograd.grad(l, x, create_graph=True)[0]
        output_grad = output_grad[:,1:] # first coordinate of x is time, so we remove it 
        
        return output, output_grad
    

# test
model = Net_Raissi(10)
x = Variable(torch.randn((2,11)), requires_grad=True)
output, output_grad = model(x)



def train():
    
    batch_size = 1000
    base_lr = 0.001
    dim = 100
    init_t = 0
    T = 1
    timestep = 0.05
    if cuda:
        timegrid = Variable(torch.Tensor(np.arange(init_t, T+timestep/2, timestep)).cuda())
    else:
        timegrid = Variable(torch.Tensor(np.arange(init_t, T+timestep/2, timestep)))
    
    sigma = math.sqrt(2)
    lambda_ = 1
    
    model = Net_Raissi(dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
    
    if cuda:
        model.cuda()
    
    n_iter = 100  # to be changed
    output_model = namedtuple('output', ['v', 'grad_v'])
    
    for it in range(n_iter):
        lr = base_lr * (0.5 ** (it // 50))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        print(it)
        model.zero_grad()
        if cuda:
            x = Variable(torch.zeros((batch_size, dim)).cuda(), requires_grad=True)
            t = Variable(torch.zeros((batch_size, 1)).cuda(), requires_grad=True)
        else:
#            x = Variable(torch.zeros((batch_size, dim)), requires_grad=True)
#            t = Variable(torch.zeros((batch_size, 1)), requires_grad=True)
             x = torch.Tensor(np.random.uniform(low=-2,high=2,size=(batch_size, dim)))
             x = Variable(x, requires_grad=True)
             t = Variable(torch.zeros((batch_size, 1)), requires_grad=True)

        tx = torch.cat([t,x], dim=1)
        
        init_time = time.time()
        v, grad_v  = model(tx) 
        end_time = time.time()
        print('time forward pass: {:.3f}'.format(end_time-init_time))
        brownian = [output_model(v, grad_v)]  # we will save the brownian path
        
        error = []
        
        for i in range(1, len(timegrid)):
            h = timegrid[i]-timegrid[i-1]
            if cuda:
                xi = Variable(torch.randn(x.size()).cuda())
            else:
                xi = Variable(torch.randn(x.size()))
            alpha = -math.sqrt(lambda_) * brownian[-1].grad_v  # to complete - make it general for any LQR problem
            f = torch.norm(alpha,2,1)**2  # to complete - make it general for any LQR problem
            if cuda:
                h = round(h.cpu().data[0],2)
                x = x + (2*math.sqrt(lambda_)*alpha) * h + sigma*math.sqrt(h)*xi
                #x = Variable(x.data, requires_grad=True)
            else:
                x = x + (2*math.sqrt(lambda_)*alpha) * h + sigma*torch.sqrt(h)*xi # to complete - make it general for any LQR problem
                #x = Variable(x.data, requires_grad=True)
            t = t+h#Variable(torch.ones((batch_size, dim)))*timegrid[i]
            t = Variable(t.data, requires_grad=True)
            tx = torch.cat([t,x], dim=1)
            #v, grad_v  = model(tx) 

            v, grad_v  = model(tx) 
            brownian.append(output_model(v, grad_v))
            if cuda:
                #h = round(h.cpu().data[0],2)
                error.append(brownian[-1].v - brownian[-2].v + 
                             f*h - 
                             torch.diag(torch.mm(brownian[-2].grad_v, sigma*math.sqrt(h)*xi.transpose(1,0))))
            else:
                error.append(brownian[-1].v - brownian[-2].v + 
                             f*h - 
                             torch.diag(torch.matmul(brownian[-2].grad_v, sigma*torch.sqrt(h)*xi.transpose(1,0))))
        # we build the loss function from Raissi's paper
        error = torch.cat(error, dim=1)
        g = torch.log(0.5*(1+torch.norm(x,2,1)**2))   # terminal condition from Raissi's paper (also from Jentzen's paper)
        g = g.view(batch_size,-1)
        error_terminal = brownian[-1].v - g
        loss = torch.sum(torch.norm(error,2,1)**2) + torch.sum(torch.norm(error_terminal,2,1)**2)
        
        # backpropagation
        init_time = time.time()
        loss.backward()
        end_time = time.time()
        print('time backpropagation: {:.3f}'.format(end_time-init_time))
        
        # Optimizer step
        optimizer.step()
        
        # printing
        print('Iteration [{it}/{n_iter}\t loss={loss:.3f}'.format(it=it, n_iter=n_iter, loss=loss.data[0]))
        
    # test
    batch_size = 1
    x = Variable(torch.zeros((batch_size, dim)).cuda(), requires_grad=True)
    t = Variable(torch.zeros((batch_size, 1)).cuda(), requires_grad=True)    
    tx = torch.cat([t,x], dim=1)
    v, grad_v  = model(tx)

        
