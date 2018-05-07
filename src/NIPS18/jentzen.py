"""
This script implements the algorithm proposed in https://arxiv.org/pdf/1706.04702.pdf
and reproduces the results, solving the HJB equation with terminal condition in the paper
"""

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce
import shutil
import time
from numpy.linalg import norm
import matplotlib.pyplot as plt
import copy
import math



#def explicit_solution_sim(x, init_t, T, lambda_, g = lambda x: 2/(1+norm(x)**2), n_simulations = 1000):
#    res_sim = []
#    h = T - init_t
#    for sim in range(n_simulations):
#        xi = np.random.normal(0,1, size=x.shape[0])
#        res = np.exp(-lambda_ * g(x+math.sqrt(2)*math.sqrt(h)*xi))
#        res_sim.append(res)
#    
#    res_sim = np.array(res_sim)
#    
#    expl_sol = -1/(lambda_) * np.log(np.mean(res_sim))
#    
#    return expl_sol, res_sim
#
#dim = 100
#x = np.zeros(dim)
#expl_sol, res_sim = explicit_solution_sim(x=x, init_t=0, T=1, lambda_=1, n_simulations=100000)


class Net_stacked(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    This network specifically solves the HJB equation with terminal condition in the paper
    
    """
    
    def __init__(self, dim, lambda_, sigma, timegrid):
        super(Net_stacked, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.lambda_ = lambda_
        self.sigma = sigma
        
        self.v0 = nn.Parameter(data=torch.randn(1))
        self.grad_v0 = nn.Parameter(data=torch.randn(self.dim))
        
        self.i_h1 = nn.ModuleList([self.hiddenLayer(dim, dim+10) for t in timegrid[1:-1]])
        self.h1_h2 = nn.ModuleList([self.hiddenLayer(dim+10, dim+10) for t in timegrid[1:-1]])
        self.h2_o = nn.ModuleList([self.outputLayer(dim+10, dim) for t in timegrid[1:-1]])
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.BatchNorm1d(nOut),
                              nn.ReLU())
        return layer
    
    def outputLayer(self, nIn, nOut):
        return nn.Linear(nIn, nOut)

    
    def forward(self, x):
        path = [x]
        dW = []
        grad_path = []
        for i in range(0,len(self.timegrid)-1):
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = Variable(torch.randn(x.data.size()))
            #print('i={}\t x.size={}\t xi.size={}'.format(i,x.data.size(),xi.data.size()))

            dW.append(xi)
            
            # we update value function
            
            if i == 0:
                grad_path.append(self.grad_v0)
                alpha = -math.sqrt(self.lambda_) * self.grad_v0
                f = torch.norm(alpha)**2
                #v = self.v0 - f*h + self.grad_v0*self.sigma*xi.transpose(0,1)
                v = self.v0 - f*h + torch.matmul(self.sigma*torch.sqrt(h)*xi, self.grad_v0)
            else:
                h1 = self.i_h1[i-1](x)
                h2 = self.h1_h2[i-1](h1)
                grad = self.h2_o[i-1](h2)
                grad_path.append(grad)
                alpha = -math.sqrt(self.lambda_) * grad
                f = torch.norm(alpha,2,1)**2
                v = v - f*h + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0)))
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            x = x + (2*math.sqrt(self.lambda_)*alpha) * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
        
        return v , x  #, path, grad_path, dW
    
    
class Net_stacked_modified(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, lambda_, sigma, timegrid):
        super(Net_stacked_modified, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.sigma = sigma
        self.lambda_ = lambda_
        
        self.v0 = nn.Parameter(data=torch.zeros(1))
        self.grad_v0 = nn.Parameter(data=torch.zeros(self.dim))
        
        self.i_h1 = nn.ModuleList([self.hiddenLayer(dim, dim+10) for t in timegrid[:-1]])
        self.h1_h2 = nn.ModuleList([self.hiddenLayer(dim+10, dim+10) for t in timegrid[:-1]])
        self.h2_o = nn.ModuleList([self.outputLayer(dim+10, dim) for t in timegrid[:-1]])
        
        self.v0_i_h1 = self.hiddenLayer(dim, dim+10)
        self.v0_h1_h2 = self.hiddenLayer(dim+10, dim+10)
        self.v0_h2_o = self.hiddenLayer(dim+10, 1)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.BatchNorm1d(nOut),
                              nn.ReLU())
        return layer
    
    def outputLayer(self, nIn, nOut):
        return nn.Linear(nIn, nOut)

    
    def forward(self, x):
        path = [x]
        dW = []
        grad_path = []
        for i in range(0,len(self.timegrid)-1):
            h1 = self.i_h1[i](x)
            h2 = self.h1_h2[i](h1)
            grad = self.h2_o[i](h2)
            grad_path.append(grad)
            alpha = -math.sqrt(self.lambda_) * grad
            f = torch.norm(alpha,2,1)**2
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = Variable(torch.randn(x.data.size()))
            
            dW.append(xi)
            
            # we update value function
            if i == 0:
                v0 = self.v0_i_h1(x)
                v0 = self.v0_h1_h2(v0)
                v0 = self.v0_h2_o(v0)
                v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            else:
                v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            x = x + alpha * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
            
        return v , x #, path #, grad_path, dW
    
    
    
def train():
    """
    
    """
    init_t, T = 0,1
    timestep = 0.05
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    dim = 100
    #g = lambda x: 2/(1+norm(x)**2)  # terminal condition value function
    #x0 = np.zeros(dim)
    sigma=math.sqrt(2) 
    lambda_ = 1

    model = Net_stacked(dim=100, lambda_=lambda_, sigma=sigma, timegrid=timegrid)

    batch_size = 1000
    base_lr = 0.01

    #optimizer = torch.optim.SGD(model.parameters(),lr=base_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
    criterion = torch.nn.MSELoss()
    
    n_iter = 1000
    v0 = []
    
    for it in range(n_iter):
        optimizer.zero_grad()
        #x0 = 10
        input = torch.zeros([batch_size, dim])  # our input is (0,0,0,...,0)
        input = Variable(input)
        output, x_T = model(input)
        #target = 2/(1+torch.norm(x_T,2,1)**2)
        target = torch.log(0.5*(1+torch.norm(x_T,2,1)**2))
        target = Variable(target.data, requires_grad=False)   # we don't want to create a loop, as alpha also depends on the parameters, and target depends on alpha
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}\t v0={v0:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0], 
              v0=copy.deepcopy(model.state_dict()['v0'].numpy())[0]))
        v0.append(copy.deepcopy(model.state_dict()['v0'].numpy())[0])
    
    return v0


def train_modified():
    init_t, T = 0,1
    timestep = 0.05
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    dim = 100
    #g = lambda x: 2/(1+norm(x)**2)  # terminal condition value function
    #x0 = np.zeros(dim)
    sigma=math.sqrt(2) 
    lambda_ = 1

    model = Net_stacked_modified(dim=100, lambda_=lambda_, sigma=sigma, timegrid=timegrid)

    batch_size = 1000
    base_lr = 0.01

    #optimizer = torch.optim.SGD(model.parameters(),lr=base_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
    criterion = torch.nn.MSELoss()
    
    n_iter = 1000
    v0 = []
    
    for it in range(n_iter):
        optimizer.zero_grad()
        #x0 = 10
        input = np.random.uniform(low=-1, high=1, size=[batch_size, dim])
        input = Variable(torch.Tensor(input))
#        input = torch.zeros([batch_size, dim])  # our input is (0,0,0,...,0)
#        input = Variable(input)
        output, x_T = model(input)
        #target = 2/(1+torch.norm(x_T,2,1)**2)
        target = torch.log(0.5*(1+torch.norm(x_T,2,1)**2))
        target = Variable(target.data, requires_grad=False)   # we don't want to create a loop, as alpha also depends on the parameters, and target depends on alpha
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
        #v0.append(copy.deepcopy(model.state_dict()['v0'].numpy())[0])
    
    return v0  


if __name__ == '__main__':
    train()
