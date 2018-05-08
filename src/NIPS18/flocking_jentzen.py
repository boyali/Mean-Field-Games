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
from scipy.interpolate import spline



# Global variable telling us whether there is GPU
cuda = False


class Net_stacked(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, kappa, sigma, law, timegrid):
        super(Net_stacked, self).__init__()
        self.dim = dim
        if cuda:
            self.timegrid = Variable(torch.Tensor(timegrid).cuda())
        else:
            self.timegrid = Variable(torch.Tensor(timegrid))
        self.law = law # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.kappa = kappa
        self.sigma = sigma
        
        self.v0 = nn.Parameter(data=torch.zeros(1))
        self.grad_v0 = nn.Parameter(data=torch.zeros(self.dim))
        
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
            if cuda:
                xi = Variable(torch.randn(x.data.size()).cuda())
                h = round(h.cpu().data[0],2)
            else:
                xi = Variable(torch.randn(x.data.size()))
            

            dW.append(xi)
            
            # we update value function
            
            if i == 0:
                grad_path.append(self.grad_v0)
                alpha = -self.grad_v0
                f = (self.kappa**2)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha)**2
                #v = self.v0 - f*h + self.grad_v0*self.sigma*xi.transpose(0,1)
                #print('x.size={}\t xi.size={}\t v0.size={}\t f.size={}'.format(x.data.size(),xi.data.size(), self.v0.data.size(), f.data.size()))
                if cuda:
                    v = self.v0 - f*h + torch.mm(self.sigma*torch.sqrt(h)*xi, self.grad_v0)
                else:
                    v = self.v0 - f*h + torch.matmul(self.sigma*torch.sqrt(h)*xi, self.grad_v0)
            else:
                h1 = self.i_h1[i-1](x)
                h2 = self.h1_h2[i-1](h1)
                grad = self.h2_o[i-1](h2)
                grad_path.append(grad)
                alpha = -grad
                f = (self.kappa**2)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha,2,1)**2
                v = v - f*h + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0)))
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            x = x + alpha * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
            
        return v , x, path #, grad_path, dW




class Net_stacked_modified(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, kappa, sigma, law, timegrid):
        super(Net_stacked_modified, self).__init__()
        self.dim = dim
        if cuda:
            self.timegrid = Variable(torch.Tensor(timegrid).cuda())
        else:
            self.timegrid = Variable(torch.Tensor(timegrid))
        self.law = law # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.kappa = kappa
        self.sigma = sigma
        
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
            alpha = -grad
            f = (self.kappa**2)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha,2,1)**2
            
            h = self.timegrid[i+1]-self.timegrid[i]
            if cuda:
                xi = Variable(torch.randn(x.data.size()).cuda())
                h = round(h.cpu().data[0],2)
            else:
                xi = Variable(torch.randn(x.data.size()))
            
            dW.append(xi)
            
            # we update value function
            if i == 0:
                v0 = self.v0_i_h1(x)
                v0 = self.v0_h1_h2(v0)
                v0 = self.v0_h2_o(v0)
                if cuda:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.mm(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            else:
                if cuda:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.mm(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            x = x + alpha * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
            
        return v , x, path #, grad_path, dW
    
    
    
class MFG_flocking():
    
    def __init__(self, dim, kappa, sigma, law, init_t, T, timestep, modified = False):
        self.dim = dim
        self.kappa = kappa
        self.sigma = sigma
        self.law = law   # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
        self.modified = modified
        
    def value_evaluation(self, batch_size=1000, base_lr=0.01, tol=0.0001, n_iter=2000):
        
        if self.modified:
            model = Net_stacked_modified(dim=self.dim, kappa=self.kappa, sigma=self.sigma, law=self.law, timegrid=self.timegrid)
        else:
            model = Net_stacked(dim=self.dim, kappa=self.kappa, sigma=self.sigma, law=self.law, timegrid=self.timegrid)
        
        if cuda:
            model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
        if cuda:
            criterion = torch.nn.MSELoss().cuda()
        else:
            criterion = torch.nn.MSELoss()
        
        model.train()
        
        v0 = []
        l = 1
        
        #for it in range(n_iter):
        it = 0
        
        while l>tol:
            optimizer.zero_grad()
            #x0 = 10
            if self.modified:
                if cuda:
                    input = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim])).cuda()
                else:
                    input = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim]))
            else:
                if cuda:
                    input = torch.ones([batch_size, dim]).cuda()  # our input is (0,0,0,...,0)
                else:
                    input = torch.ones([batch_size, dim])
            input = Variable(input)
            output, x_T, _ = model(input)
            #target = 2/(1+torch.norm(x_T,2,1)**2)
            #target = torch.log(0.5*(1+torch.norm(x_T,2,1)**2))
            #target = Variable(target.data, requires_grad=False)   # we don't want to create a loop, as alpha also depends on the parameters, and target depends on alpha
            if cuda:
                target = Variable(torch.zeros([batch_size,1]).cuda())  # terminal condition in Flocking model is 0
            else:
                target = Variable(torch.zeros([batch_size,1]))  # terminal condition in Flocking model is 0
                
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()
            if self.modified:
                print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            else:
                print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}\t v0={v0:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0], 
                  v0=copy.deepcopy(model.state_dict()['v0'].numpy())[0]))
                v0.append(copy.deepcopy(model.state_dict()['v0'].numpy())[0])
            l = loss.data[0]
            it += 1
        print('Its over!!!')
            
        return model#, v0
        
    def law_improvement(self, model, n_iter=10000):
            
        model.eval()
        input = Variable(torch.ones([n_iter, self.dim]))
        _, _, path = model(input)
        improved_law = []
        std_law = []
        for step in path:
            improved_law.append(step.mean(0).view(1,-1))
            std_law.append(step.std(0).view(1,-1))
        self.law = torch.cat(improved_law, 0)
        self.law = Variable(self.law.data)
        self.std_law = torch.cat(std_law,0)
        
    def law_improvement_modified(self, model, init_values, n_iter = 10000):
        model.eval()
        improved_law = []
        for player in range(init_values.size()[0]):
            input = Variable(torch.cat([init_values[player]*n_iter], dim=0))
            _, _, path = model(input)
            for step in path:
                improved_law.append(step.mean(0).view(1,-1))
        self.law = torch.cat(improved_law, 0)
        self.law = Variable(self.law.data)
        
    
    
    def get_policy(self, model, x):
        """
        This function returns the policy, that we know is alpha=-grad
        """
        model.eval()
        alpha = torch.zeros([len(self.timegrid)-1, self.dim])
        for t in range(len(self.timegrid)-1):
            if self.modified:
                h1 = model.i_h1[i](x)
                h2 = model.h1_h2[i](h1)
                grad = model.h2_o[i](h2)
                alpha[t] = -grad
            else:
                if t==0:
                    alpha[t] = -model.grad_v0
                else:
                    h1 = model.i_h1[i-1](x)
                    h2 = model.h1_h2[i-1](h1)
                    grad = model.h2_o[i-1](h2)
                    alpha[t] = -grad
        return alpha
            
        


class flocking_model():
    
    def __init__(self, N=100, h=1, kappa=1, sigma=0.01, T=100, dim=3):
        self.N = N
        self.h = h
        self.kappa = kappa
        self.sigma = sigma
        self.T = T
        self.dim = dim
        self.states = self.init_markov()
        self.alphas = np.zeros((self.N, int(self.T/self.h)-1))
        self.Wiener = np.zeros((self.N, int(self.T/self.h)-1))
        self.etas = np.zeros(int(self.T/self.h)-1)
        

    def eta(self, t):
        val = self.kappa*(math.sqrt(self.N/(self.N-1)))
        val = val*math.tanh(self.kappa*math.sqrt((self.N - 1)/self.N)*(self.T - t*self.h))
        self.etas[int(t)] = val
        return val
    
    def next_state(self, x, t):
        xi = np.random.normal(loc=0, scale=1, size=self.dim)
        mean_states_t = np.apply_along_axis(np.mean, 0, self.states[:,t,:])
        next_x = x - self.eta(t)*(1-1/self.N)*(x-mean_states_t)*self.h + self.sigma*xi*math.sqrt(self.h)
        return next_x
    
    def init_markov(self):
        states = np.zeros((self.N, int(self.T/self.h),self.dim))
        #states[:,0] = np.random.normal(loc=0, scale=2, size=(self.N,self.dim))
        return states
    
#    def get_alpha(self,i,t):
#        val = -1*(1-1/self.N)*self.eta(self.h*t)*(self.states[i,t]-np.mean(self.states[:,t]))
#        self.alphas[i,t] = val
#        return val
    
    def simulate_mdp(self):
        for t in range(1, int(self.T/self.h)):
            for i in range(self.N):
                #noise = self.sigma*random.gauss(0,1)
                self.states[i,t] = self.next_state(self.states[i,t-1], t-1)        
        
        

def game_Jentzen_Flocking():
    
    dim = 10
    kappa = 1
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.01
    
    # EXPLICIT SOLUTON OF FLOCKING MODEL FROM MFG BOOK (eq. 2.51)   
    flocking_mdp = flocking_model(N=100, h=0.05, kappa=1, sigma=0.01, T=10, dim=3)
    flocking_mdp.simulate_mdp()
    
    law = []
    for i in range(flocking_mdp.states.shape[1]):
        law.append(np.apply_along_axis(np.mean, 0, flocking_mdp.states[:,i,:]))
    law = np.concatenate(law, axis=0)
    
    
    
    # PLOTTING. We plot one of the dimensions
    test = flocking_mdp.states[:,:,0]
    test = test.reshape(test.shape[0],test.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in range(test.shape[0]):
        ax.plot(test[row,:])
    
    
    # DEEP LEARNING SOLUTION
    #dim = 100
    dim = 100
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    #law = Variable(torch.zeros([timegrid.size, dim]))
    law = np.random.normal(loc=1, scale=0.1, size=[timegrid.size, dim]) # the law is drawn from a normal distribution with mean 1, sd=0.1
    if cuda:
        law = Variable(torch.Tensor(law).cuda())
    else:
        law = Variable(torch.Tensor(law))
    
    game = MFG_flocking(dim=dim, kappa=1, sigma=0.01, law=law, init_t=0, T=1, timestep=timestep)
    law = [game.law]
    models = []
    model = game.value_evaluation(n_iter=1500, base_lr=0.01, tol=0.0006)
    game.law_improvement(model, n_iter=30000)
    game.law
    law.append(game.law)
    models.append(model)
    
    game.std_law
    


def game_Jentzen_Flocking_extended():
    
    dim = 10
    kappa = 1
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.01
    
    # EXPLICIT SOLUTON OF FLOCKING MODEL FROM MFG BOOK (eq. 2.51)   
    flocking_mdp = flocking_model(N=100, h=0.05, kappa=1, sigma=0.01, T=10, dim=3)
    flocking_mdp.simulate_mdp()
    
    law = []
    for i in range(flocking_mdp.states.shape[1]):
        law.append(np.apply_along_axis(np.mean, 0, flocking_mdp.states[:,i,:]))
    law = np.concatenate(law, axis=0)
    
    
    
    # PLOTTING. We plot one of the dimensions
    test = flocking_mdp.states[:,:,0]
    test = test.reshape(test.shape[0],test.shape[1])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in range(test.shape[0]):
        ax.plot(test[row,:])
    
    
    # DEEP LEARNING SOLUTION
    #dim = 100
    dim = 10
    timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    if cuda:
        law = Variable(torch.zeros([timegrid.size, dim]).cuda())
    else:
        law = Variable(torch.zeros([timegrid.size, dim]))
    #law = np.random.normal(loc=0, scale=0.01, size=[timegrid.size, dim]) # the law is drawn from a normal distribution with mean 1, sd=0.1
    #law = Variable(torch.Tensor(law))
    
    game = MFG_flocking(dim=dim, kappa=1, sigma=0.01, law=law, init_t=0, T=1, timestep=timestep, modified=True)
    law = [game.law]
    model, v0 = game.value_evaluation(n_iter=1500, base_lr=0.01, tol=0.00001)
    game.law_improvement(model, n_iter=30000)
    game.law
    law.append(game.law)
    
    game.std_law
    
    
            
            
        
        
            
        
        
        
        
        
        
        
    