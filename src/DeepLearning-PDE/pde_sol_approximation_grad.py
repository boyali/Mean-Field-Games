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



#We do a first example wit alpha = 0.5*x

       
class Net_stacked(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the hamiltonian equation with terminal condition
    v(x,T) = gamma * x**2
    For this, we fix alpha = 0.5x as an example
    If alpha is not fixed, and we want to find the optimal the optimal strategy,
    we can still apply the same algorithm by solving a non-linear PDE. 
    
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    """
    
    def __init__(self, dim, b, c, sigma, b_f, c_f, timegrid):
        super(Net_stacked, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.v0 = nn.Parameter(data=torch.randn([self.dim]))
        self.grad_v0 = nn.Parameter(data=torch.randn([dim]))
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
        
        for i in range(0,len(self.timegrid)-1):
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = torch.sqrt(h)*Variable(torch.randn(x.data.size()))
            #print('i={}\t x.size={}\t xi.size={}'.format(i,x.data.size(),xi.data.size()))
            alpha = -0.5*x + 0.1
            
            
            # we update value function
            if i == 0:
                v = self.v0 - (self.b_f*x**2 + self.c_f*alpha**2)*h + self.grad_v0*self.sigma*xi
            else:
                h1 = self.i_h1[i-1](x)
                h2 = self.h1_h2[i-1](h1)
                grad = self.h2_o[i-1](h2)

                v = v - (self.b_f*x**2 + self.c_f*alpha**2)*h + grad*self.sigma*xi
            
            # we update x
            x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            #print('x={}, sigma={}, alpha={}, xi={}'.format(x.data, self.sigma, alpha.data, xi.data))
        
        return v, x
    
    
class Net_stacked_modified(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    v(x,T) = gamma * x**2
    The modification of this network is that it won't only calculate v(0,x0) for a fixed x0, but v(0,x) for any x
    
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    """
    
    def __init__(self, dim, b, c, sigma, b_f, c_f, timegrid):
        super(Net_stacked_modified, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        #self.v0 = nn.Parameter(data=torch.randn([self.dim]))
        #self.grad_v0 = nn.Parameter(data=torch.randn([dim]))
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
        
        for i in range(0,len(self.timegrid)-1):
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = torch.sqrt(h)*Variable(torch.randn(x.data.size()))
            #print('i={}\t x.size={}\t xi.size={}'.format(i,x.data.size(),xi.data.size()))
            alpha = -0.5*x + 0.1
            
            h1 = self.i_h1[i-1](x)
            h2 = self.h1_h2[i-1](h1)
            grad = self.h2_o[i-1](h2)
            
            # we update value function
            if i == 0:
                v0 = self.v0_i_h1(x)
                v0 = self.v0_h1_h2(v0)
                v0 = self.v0_h2_o(v0)
                v = v0 - (self.b_f*x**2 + self.c_f*alpha**2)*h + grad*self.sigma*xi
            else:
                v = v - (self.b_f*x**2 + self.c_f*alpha**2)*h + grad*self.sigma*xi
            
            # we update x
            x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
        return v, x
    
       

def train():
    init_t, T = 0,1
    timestep = 0.01
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    x0 = 10
    b=0.5 
    c=0.5 
    sigma=1 
    b_f=0.5 
    c_f=0.9
    gamma = 1
    
    model = Net_stacked(dim=1, b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, timegrid=timegrid)
    batch_size = 120
    base_lr = 0.2
    #optimizer = torch.optim.SGD(model.parameters(),lr=base_lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
    criterion = torch.nn.MSELoss()
    
    n_iter = 3000
    v0 = []
    
    for it in range(n_iter):
        optimizer.zero_grad()
        x0 = 10
        input = torch.ones([batch_size, 1])*x0
        input = Variable(input)
        output, x_T = model(input)
        target = gamma*x_T**2
        #target = Variable(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
        v0.append(copy.deepcopy(model.state_dict()['v0'].numpy())[0])
    
    return v0


def train_modified():
    init_t, T = 0,1
    timestep = 0.01
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    b=0.5
    c=0.5
    sigma=1
    b_f=0.5
    c_f=0.9
    gamma = 1
    
    model= Net_stacked_modified(dim=1, b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, timegrid=timegrid)
    batch_size = 60
    base_lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    criterion = torch.nn.MSELoss()
    
    n_iter = 3000
    #v0 = []
    
    for it in range(n_iter):
        optimizer.zero_grad()
        input = torch.rand([batch_size,1])
        input = Variable(input)
        output,x_T = model(input)
        target = gamma*x_T**2
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
        

if __name__=='__main__':
    train()


    
##### PLOTTING 
import matplotlib.pyplot as plt
v0 = np.array(v0)
iteration = np.arange(1,len(v0)+1)
fig =plt.figure()
ax = fig.add_subplot(111)
ax.plot(iteration, v0, 'b-')
ax.set_xlabel('iteration')
ax.set_ylabel('v(0,10)')
plt.show()
fig.savefig('DL_v0.png')
        
        
        