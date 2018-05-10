import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from functools import reduce
import time



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cuda = False


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
        
        l = [output[row] for row in range(output.size()[0])]
        output_grad = torch.autograd.grad(l, x, create_graph=True)[0]
        output_grad = output_grad[:,1:] # first coordinate of x is time, so we remove it 
        
        return output, output_grad




class MFG_flocking():
    
    def __init__(self, dim, kappa, sigma, law, init_t, T, timestep):
        self.dim = dim
        self.kappa = kappa
        self.sigma = sigma
        self.law = law   # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        #self.timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
        if cuda:
            self.timegrid = Variable(torch.Tensor(np.arange(init_t, T+timestep/2, timestep)).cuda())
        else:
            self.timegrid = Variable(torch.Tensor(np.arange(init_t, T+timestep/2, timestep)))

        
    def value_evaluation(self, batch_size=100, base_lr=0.001, tol=1, n_iter=2000):
        """
        Training of the model Net_raissi.
        """
        model = Net_Raissi(dim = self.dim)
        
        if cuda:
            model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(),lr=base_lr)
        output_model = namedtuple('output', ['v', 'grad_v'])
        
        #model.train()
        
        for it in range(n_iter):
            lr = base_lr * (0.5 ** (it // 50))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
            #print(it)
            model.zero_grad()
            if cuda:
                x = torch.Tensor(np.random.uniform(low=-1,high=1,size=(batch_size, self.dim))).cuda()
                x = Variable(x, requires_grad=True)
                #x = Variable(torch.zeros((batch_size, self.dim)).cuda(), requires_grad=True)
                t = Variable(torch.zeros((batch_size, 1)).cuda(), requires_grad=True)
            else:
                #x = Variable(torch.zeros((batch_size, self.dim)), requires_grad=True)
                x = torch.Tensor(np.random.uniform(low=-1,high=1,size=(batch_size, self.dim)))
                x = Variable(x, requires_grad=True)
                t = Variable(torch.zeros((batch_size, 1)), requires_grad=True)
            
            tx = torch.cat([t,x], dim=1)
            
            init_time = time.time()
            v, grad_v  = model(tx) 
            end_time = time.time()
            #print('time forward pass: {:.3f}'.format(end_time-init_time))
            
            brownian = [output_model(v, grad_v)]  # we will save the brownian path
            
            error = []
            
            for i in range(1, len(self.timegrid)):
                h = self.timegrid[i]-self.timegrid[i-1]
                if cuda:
                    xi = Variable(torch.randn(x.size()).cuda())
                else:
                    xi = Variable(torch.randn(x.size()))
                alpha = -grad_v  # to complete - make it general for any LQR problem
                
                init_time = time.time()
                f = (self.kappa**2)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha,2,1)**2
                end_time = time.time()
                #print('time to calculate f: {:.3f}'.format(end_time-init_time))
                if cuda:
                    h = round(h.cpu().data[0],2)
                    x = x + (alpha) * h + self.sigma*math.sqrt(h)*xi
                    x = Variable(x.data, requires_grad=True)
                else:
                    init_time = time.time()
                    x = x + (alpha) * h + self.sigma*torch.sqrt(h)*xi # to complete - make it general for any LQR problem
                    x = Variable(x.data, requires_grad=True)
                    end_time = time.time()
                    #print('time to make brownian step: {:.3f}'.format(end_time-init_time))
                
                t = t+h#Variable(torch.ones((batch_size, dim)))*timegrid[i]
                t = Variable(t.data, requires_grad=True)
                tx = torch.cat([t,x], dim=1)
                #v, grad_v  = model(tx) 
    
                init_time = time.time()
                v, grad_v  = model(tx) 
                end_time = time.time()
                #print('time forward pass: {:.3f}'.format(end_time-init_time))
                brownian.append(output_model(v, grad_v))
                if cuda:
                    #h = round(h.cpu().data[0],2)
                    error.append(brownian[-1].v - brownian[-2].v + 
                                 f*h - 
                                 torch.diag(torch.mm(brownian[-2].grad_v, self.sigma*math.sqrt(h)*xi.transpose(1,0))))
                else:
                    error.append(brownian[-1].v - brownian[-2].v + 
                                 f*h - 
                                 torch.diag(torch.matmul(brownian[-2].grad_v, self.sigma*torch.sqrt(h)*xi.transpose(1,0))))
            # we build the loss function from Raissi's paper
            error = torch.cat(error, dim=1)
            #g = torch.log(0.5*(1+torch.norm(x,2,1)**2))   # terminal condition from Raissi's paper (also from Jentzen's paper)
            #g = g.view(batch_size,-1)
            #error_terminal = brownian[-1].v - g
            error_terminal = brownian[-1].v
            loss = torch.sum(torch.norm(error,2,1)**2) + torch.sum(torch.norm(error_terminal,2,1)**2)
            
            # backpropagation
            init_time = time.time()
            loss.backward()
            end_time = time.time()
            #print('time backpropagation: {:.3f}'.format(end_time-init_time))
            
            # Optimizer step
            init_time = time.time()
            optimizer.step()
            end_time = time.time()
            #print('time optimizer step: {:.3f}'.format(end_time-init_time))
            
            # printing
            print('Iteration [{it}/{n_iter}\t loss={loss:.3f}'.format(it=it, n_iter=n_iter, loss=loss.data[0]))
        
            
        return model#, v0
        
    
    def _get_path(self, model, x):
        """
        Input: 
            - x. Matrix batch_size x dim. All the rows should be the same
        Output: 
            - path. Matrix len(timegrid) x dim. Each vector is a vector of length dim. 
        """
        model.eval()
        if cuda:
            t = Variable(torch.zeros((x.size()[0], 1)).cuda(), requires_grad=True)
        else:
            t = Variable(torch.zeros((x.size()[0], 1)), requires_grad=True)
        tx = torch.cat([t,x], dim=1)
        v, grad_v = model(tx)
        path = [x.data]
        
        for i in range(1, len(self.timegrid)):
            h = self.timegrid[i]-self.timegrid[i-1]
            if cuda:
                xi = Variable(torch.randn(x.size()).cuda())
            else:
                xi = Variable(torch.randn(x.size()))
            alpha = -grad_v  # to complete - make it general for any LQR problem
            if cuda:
                h = round(h.cpu().data[0],2)
                x = x + (alpha) * h + self.sigma*math.sqrt(h)*xi
                x = Variable(x.data, requires_grad=True)
            else:
                x = x + (alpha) * h + self.sigma*torch.sqrt(h)*xi # to complete - make it general for any LQR problem
                x = Variable(x.data, requires_grad=True)
            
            path.append(x.data)
        path = [step.mean(0).view(1,-1) for step in path]
        path = torch.cat(path, 0)
        return path 

        
    def law_improvement(self, model, init_values, n_iter=1000):
        """
        Input: 
            - init_values: a matrix n_players x dim with the init_values of the n_players
        Output: 
            - self.law: a matrix len(timegrid) x dim with the law at each time step 
        """
        sims = []
        for player in range(init_values.size()[0]):
            print('player {}'.format(player))
            x = init_values[player].view(1,-1)
            for it in range(n_iter):
                sims.append(self._get_path(model, x))
        self.law = reduce(lambda x,y: x+y, sims)
        self.law = self.law/(len(sims))
        self.law = Variable(self.law)
        return sims
        
    def get_policy(self, model, x):
        """
        This function returns the policy, that we know is alpha=-grad
        Input:
            - x: vector of size (1,dim), that we will use to get the policy for all t in timegrid
        Ouput:
            - alpha: matrix len(timegrid) x dim: alpha(t,x) for each t in timegrid
        """
        model.eval()
        alpha = torch.zeros([len(self.timegrid)-1, self.dim])
        for i in range(len(self.timegrid)-1):
            t = self.timegrid[i].view(1,1)            
            tx = torch.cat([t,x], dim=1)
            v, grad_v = model(tx)
            alpha[i] = grad_v
        return alpha
    
    
def solve_MFG():
    """
    This script solves the Flocking model using Raissi's algorithm and Monte Carlo simulations
    """
    dim = 10
    kappa = 1
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    n_players = 10
                         
    law = torch.ones(timegrid.size, dim)*0.5
    if cuda:
        law = Variable(torch.Tensor(law).cuda())
    else:
        law = Variable(torch.Tensor(law))
        
    game = MFG_flocking(dim=dim, kappa=kappa, sigma=sigma, law=law, init_t=init_t, T=T, timestep=timestep)
    laws = [game.law]
    if cuda:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim])).cuda()
    else:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim]))
    init_values = Variable(init_values)


    models = []

    model = game.value_evaluation(n_iter=100, base_lr=0.001, tol=0.0001, batch_size = 200)
    sims = game.law_improvement(model, n_iter=1000, init_values = init_values)
    game.law
    laws.append(game.law)
    models.append(model)
    
    
    
    norms = []
    for i in range(1, len(laws)):
        a = (laws[i]-laws[i-1]).cpu().data.numpy()
        norms.append(norm(a))
    
    x = Variable(torch.zeros(1, dim))
    game.get_policy_modified(models[-1], x)


        


    


    