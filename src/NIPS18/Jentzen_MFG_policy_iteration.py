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
#import matplotlib.pyplot as plt
import copy
import math
from scipy.interpolate import spline
import random

cuda = True


class Net_alpha(nn.Module):
    """
    We create the alpha net
    """
    
    def __init__(self, dim):
        super(Net_alpha, self).__init__()
        self.dim = dim
        
        self.i_h1 = self.hiddenLayer(dim+1, dim+100)
        self.h1_h2 = self.hiddenLayer(dim+100, dim+100)
        #self.h2_h3 = self.hiddenLayer(dim+100, dim+100)
        #self.h3_h4 = self.hiddenLayer(dim+100, dim+100)
        #self.h4_h5 = self.hiddenLayer(dim+100, dim+100)
        #self.h5_o = nn.Linear(dim+100, dim)
        self.h2_o = nn.Linear(dim+100, dim)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut),
                              nn.BatchNorm1d(nOut),
                              nn.ReLU())
        return layer
    
    def forward(self, tx):
        h1 = self.i_h1(tx)
        h2 = self.h1_h2(h1)
        #h3 = self.h2_h3(h2)
        #h4 = self.h3_h4(h3)
        #h5 = self.h4_h5(h4)
        #alpha = self.h5_o(h5)
        alpha = self.h2_o(h2)
        return alpha
#    
#class Net_alpha(nn.Module):
#    def __init__(self, dim):
#        super(Net_alpha, self).__init__()
#        self.dim = dim
#        
#        self.i_h1 = nn.Sequential(nn.Linear(dim+1, 100), nn.ReLU())
#        self.i_h2 = nn.Sequential(nn.Linear(dim+1, 100), nn.Tanh())
#        self.h3_o = nn.Linear(100,dim)
#    
#    def forward(self,tx):
#        h1 = self.i_h1(tx)
#        h2 = self.i_h2(tx)
#        h3 = h1*h2
#        o = self.h3_o(h3)
#        return o



#class Net_alpha(nn.Module):
#    def __init__(self, dim):
#        super(Net_alpha, self).__init__()
#        self.dim = dim
#        
#        self.i_h1 = nn.Sequential(nn.Linear(dim+1, 50), nn.ReLU())
#        self.i_h2 = nn.Sequential(nn.Linear(dim+1, 50), nn.Tanh())
#        self.i_h4 = nn.Sequential(nn.Linear(dim+1, 50), nn.ReLU())
#        self.h4_o = nn.Linear(100,dim)
#    
#    def forward(self,tx):
#        h1 = self.i_h1(tx)
#        h2 = self.i_h2(tx)
#        h3 = h1*h2
#        h4 = self.i_h4(tx)
#        h4 = torch.cat([h3,h4],1)
#        o = self.h4_o(h4)
#        return o

class Net_Jentzen_1network(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, kappa, sigma, law, timegrid):
        super(Net_Jentzen_1network, self).__init__()
        self.dim = dim
        if cuda:
            self.timegrid = torch.Tensor(timegrid).cuda()
        else:
            self.timegrid = torch.Tensor(timegrid)
        self.law = law # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.kappa = kappa
        self.sigma = sigma
        netAlpha.eval()
        
        self.i_h1 = self.hiddenLayer(dim+1, dim+100)  # dim+1 because we have time
        self.h1_h2 = self.hiddenLayer(dim+100, dim+100)
        self.h2_o = nn.Linear(dim+100, dim)  
                
        self.v0_i_h1 = self.hiddenLayer(dim, dim+100)
        self.v0_h1_h2 = self.hiddenLayer(dim+100, dim+100)
        self.v0_h2_o = nn.Linear(dim+100, 1)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.BatchNorm1d(nOut), # if we use automatic differentiation we cannot use BatchNorm1d
                              nn.ReLU())
        return layer
    
    def outputLayer(self, nIn, nOut):
        return nn.Linear(nIn, nOut)

    
    def forward(self, x):
        path = [x]
        dW = []
        for i in range(0,len(self.timegrid)-1):
                        #we calculate the alpha
            t = torch.ones(x.size()[0], 1)*self.timegrid[i]
            if cuda:
                t = Variable(t.cuda())
            else:
                t = Variable(t)
            tx = torch.cat([t,x],1)
            alpha_tx = netAlpha(tx)
            h1 = self.i_h1(tx)
            h2 = self.h1_h2(h1)
            grad = self.h2_o(h2)
            
            f = (self.kappa)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
            #f = -(self.kappa)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha,2,1)**2
            
            h = self.timegrid[i+1]-self.timegrid[i]
            if cuda:
                xi = Variable(torch.randn(x.data.size()).cuda())
                #h = round(h.cpu().data[0],2)
            else:
                xi = Variable(torch.randn(x.data.size()))
            
            dW.append(xi)
            
            # we update value function
            if i == 0:
                v0 = self.v0_i_h1(x)
                v0 = self.v0_h1_h2(v0)
                v0 = self.v0_h2_o(v0)
                if cuda:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*math.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            else:
                if cuda:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*math.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            if cuda:
                x = x + alpha_tx * h + self.sigma*math.sqrt(h)*xi
            else:
                x = x + alpha_tx * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
        return v , x, path #, grad_path, dW
        




class Net_Jentzen(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, kappa, sigma, law, timegrid):
        super(Net_Jentzen, self).__init__()
        self.dim = dim
        if cuda:
            self.timegrid = torch.Tensor(timegrid).cuda()
        else:
            self.timegrid = torch.Tensor(timegrid)
        self.law = law # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.kappa = kappa
        self.sigma = sigma
        netAlpha.eval()
                
        self.i_h1 = nn.ModuleList([self.hiddenLayer(dim, dim+10) for t in timegrid[:-1]])
        self.h1_h2 = nn.ModuleList([self.hiddenLayer(dim+10, dim+10) for t in timegrid[:-1]])
        self.h2_o = nn.ModuleList([self.outputLayer(dim+10, dim) for t in timegrid[:-1]])
        
        self.v0_i_h1 = self.hiddenLayer(dim, dim+10)
        self.v0_h1_h2 = self.hiddenLayer(dim+10, dim+10)
        self.v0_h2_o = self.hiddenLayer(dim+10, 1)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.BatchNorm1d(nOut), # if we use automatic differentiation we cannot use BatchNorm1d
                              nn.ReLU())
        return layer
    
    def outputLayer(self, nIn, nOut):
        return nn.Linear(nIn, nOut)

    
    def forward(self, x):
        path = [x]
        dW = []
        for i in range(0,len(self.timegrid)-1):
            h1 = self.i_h1[i](x)
            h2 = self.h1_h2[i](h1)
            grad = self.h2_o[i](h2)
            
            #we calculate the alpha
            t = torch.ones(x.size()[0], 1)*self.timegrid[i]
            if cuda:
                t = Variable(t.cuda())
            else:
                t = Variable(t)
            tx = torch.cat([t,x],1)
            alpha_tx = netAlpha(tx)
            
            f = (self.kappa)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
            #f = -(self.kappa)/2 * torch.norm(x-self.law[i],2,1)**2 +  0.5*torch.norm(alpha,2,1)**2
            
            h = self.timegrid[i+1]-self.timegrid[i]
            if cuda:
                xi = Variable(torch.randn(x.data.size()).cuda())
                #h = round(h.cpu().data[0],2)
            else:
                xi = Variable(torch.randn(x.data.size()))
            
            dW.append(xi)
            
            # we update value function
            if i == 0:
                v0 = self.v0_i_h1(x)
                v0 = self.v0_h1_h2(v0)
                v0 = self.v0_h2_o(v0)
                if cuda:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*math.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v0 - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            else:
                if cuda:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*math.sqrt(h)*xi.transpose(1,0))).view(-1,1)
                else:
                    v = v - (f*h).view(-1,1) + torch.diag(torch.matmul(grad, self.sigma*torch.sqrt(h)*xi.transpose(1,0))).view(-1,1)
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            if cuda:
                x = x + alpha_tx * h + self.sigma*math.sqrt(h)*xi
            else:
                x = x + alpha_tx * h + self.sigma*torch.sqrt(h)*xi
            path.append(x)
        return v , x, path #, grad_path, dW
    



def evaluate_value_function(kappa, sigma, timegrid, law, index_t, x,netJentzen, netAlpha):
    """
    We evaluate the value function at a point (t,x) using the following algorithm:
        - we build a random path using dXt = alpha*dt + sigma*dWt
        - We keep the dWt
        - We go backwards from g(X_T) to v(t,x) using Euler-Maruyama
    Input:
        - index_t: index of time in timegrid
        - x. Matrix of size (batch_size, dim)
        - Jentzen trained network
        - alpha network
    Output:
        - v(t,x): Matrix of size (batch_size, 1)
    """
    netJentzen.eval()
    #law.eval()
    
    if index_t == 0:
        v0 = netJentzen.v0_i_h1(x)
        v0 = netJentzen.v0_h1_h2(v0)
        v0 = netJentzen.v0_h2_o(v0)
        return v0
    else:
        # 1. We get states path, and save dW
        dW = []
        path = [x]
        alpha = []
        for i in range(index_t, len(timegrid)-1):
            t = torch.ones(x.size()[0], 1)*timegrid[i]
            if cuda:
                t = Variable(t.cuda())
            else:
                t = Variable(t)
            tx = torch.cat([t,x],1)
            alpha_tx = netAlpha(tx)
            alpha.append(alpha_tx)
            h = timegrid[i+1]-timegrid[i]
            if cuda:
                xi = Variable(torch.randn(x.data.size()).cuda())
                h = round(h.cpu().data[0],2)
            else:
                xi = Variable(torch.randn(x.data.size()))        
            dW.append(xi)
            if cuda:
                x = x + alpha_tx * h + sigma*math.sqrt(h)*xi
            else:
                x = x + alpha_tx * h + sigma*torch.sqrt(h)*xi
            path.append(x)
        
        # 2. We go backwards and get v(t,x)
        terminal_condition = torch.zeros(x.size()[0],1)
        if cuda:
            terminal_condition = Variable(terminal_condition.cuda())
        else:
            terminal_condition = Variable(terminal_condition)
        v = terminal_condition
        index_path = -1
        for i in range(len(timegrid), index_t, -1):
            
            grad = netJentzen.i_h1[i](x)
            grad = netJentzen.h1_h2[i](grad)
            grad = netJentzen.h2_o[i](grad)
            
            h = timegrid[i+1]-timegrid[i]
            if cuda:
                h = round(h.cpu().data[0],2)
            
            f = (kappa)/2 * torch.norm(path[index_path]-law[i],2,1)**2 +  0.5*torch.norm(alpha[index_path],2,1)**2
            v = v + (f*h).view(-1,1) - torch.diag(torch.matmul(grad, sigma*torch.sqrt(h)*dW[index_path].transpose(1,0))).view(-1,1)
            
            index_path = index_path-1
        return v



def get_hessian_automatic_differentiation(index_t, x,netJentzen):
    """
    This function gets trace(\delta_xx v )
    Input:
        - index_t: index of time within timegrid
        - x: matrix of shape (batch_size, dim)
    Output:
        - tr(hessian): trace of the hessian. Matrix of shape (batch_size, 1) 
    """
    netJentzen.eval()
    
    grad = netJentzen.i_h1[index_t](x)
    grad = netJentzen.h1_h2[index_t](grad)
    grad = netJentzen.h2_o[index_t](grad)   # grad is a matrix of shape (batch_size, dim) 
    
    diag_hess = []
    for col in range(x.size()[1]):  # each col is a dimension
        l = [grad[row][col] for row in range(grad.size()[0])]
        hess = torch.autograd.grad(l, x, create_graph=True)[0]
        diag_hess.append(hess[:,col].contiguous().view(-1,1))
    diag_hess = torch.cat(diag_hess, 1)
    tr_hess = diag_hess.sum(1)
    return tr_hess




def get_hessian_finite_diff(index_t, x, netJentzen):
    """
    This function gets trace(\delta_xx v)
    Input:
        - index_t: index of time within timegrid
        - x: matrix of shape (batch_size, dim)
    Output:
        - tr(hessian): trace of the hessian. Matrix of shape (batch_size, 1), each row being the tr(hessian) of the correpsonding point in the batch of data
    """
    netJentzen.eval()
    
    grad = netJentzen.i_h1[index_t](x)
    grad = netJentzen.h1_h2[index_t](grad)
    grad = netJentzen.h2_o[index_t](grad)   # grad is a matrix of shape (batch_size, dim) 
    x.requires_grad=False
    h = 0.001
    diag_hess = []
    for col in range(x.size()[1]):
        y = copy.deepcopy(x)
        y[:,col] = y[:,col]+h
        grad_h = netJentzen.i_h1[index_t](y)
        grad_h = netJentzen.h1_h2[index_t](grad_h)
        grad_h = netJentzen.h2_o[index_t](grad_h)   # grad is a matrix of shape (batch_size, dim) 
        hess =(grad_h-grad)/h
        diag_hess.append(hess[:,col].contiguous().view(-1,1))
    diag_hess_doff = torch.cat(diag_hess,1)
    tr_hess_diff = diag_hess_doff.sum(1)
    return tr_hess_diff
    
    
    
def evaluation_step(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep, law):
    """
    This function solves the PDE, with fixed law and alpha (control)
    Inputs:
        - kappa, sigma: floats
        - timegrid: 
    """
    #timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
    #model = Net_Jentzen(dim=dim, kappa=kappa, sigma=sigma, alpha=alpha, law=law, timegrid=timegrid)
    if cuda:
        netJentzen.cuda()
    
    optimizer = torch.optim.Adam(netJentzen.parameters(),lr=base_lr)
    if cuda:
        criterion = torch.nn.MSELoss().cuda()
    else:
        criterion = torch.nn.MSELoss()
    
    netJentzen.train()
    
    it = 0
    for it in range(n_iter):
        
        # parameter decay
        lr = base_lr * (0.5 ** (it // 50))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        if cuda:
            input = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim])).cuda()
        else:
            input = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim]))
        input = Variable(input)
        init_time = time.time()
        output, x_T, _ = netJentzen(input)
        end_time = time.time()
        #print('time forward pass: {:.4f}'.format(end_time-init_time))
        if cuda:
            target = Variable(torch.zeros([batch_size,1]).cuda())  # terminal condition in Flocking model is 0
        else:
            target = Variable(torch.zeros([batch_size,1]))  # terminal condition in Flocking model is 0
            
        loss = criterion(output, target) 
        init_time = time.time()
        loss.backward()
        end_time = time.time()
        #print('time backward propagation: {:.4f}'.format(end_time-init_time))     
        init_time = time.time()
        optimizer.step()
        end_time = time.time()
        #print('time backward optimization step: {:.4f}'.format(end_time-init_time))     
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
    
    print('Its over!!!')
    #return model
    
    

def policy_improvement_step_old(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep, law):
    
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #model = Net_alpha(dim)
    netAlpha.train()
    netJentzen.eval()
    
    optimizer = torch.optim.Adam(netAlpha.parameters(),lr=base_lr)
    optimizer = torch.optim.SGD(netAlpha.parameters(),lr=base_lr, momentum=0.9, nesterov=True)
    
    for it in range(n_iter):
        # optimisation step decay
        #for param_group in optimizer.state_dict()['param_groups']:
        #lr = base_lr * (0.5 ** (it // 1000))
        #    param_group['lr'] = lr
        optimizer.zero_grad()        
        if cuda:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1, dim])).cuda()
        else:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1, dim]))
        x = Variable(x)
        index_t = random.randint(0,len(timegrid)-2)
        t = torch.ones(1,1)*timegrid[index_t]
        if cuda:
            t = Variable(t.cuda())
        else:
            t = Variable(t)
        tx = torch.cat([t,x], 1)
        alpha_tx = netAlpha(tx.view(1,-1))
        #f = (kappa)/2 * torch.norm(x-law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
        f = 0.5*torch.norm(alpha_tx,2,1)**2 # we just take the part of f where alpha is involved
        f = f.view(-1,1)
        #tr_hess = get_hessian_finite_diff(index_t=i, x=x, netJentzen=netJentzen)
        #tr_hess = tr_hess.view(-1,1)
        
        grad = netJentzen.i_h1[index_t](x.view(1,-1))
        grad = netJentzen.h1_h2[index_t](grad)
        grad = netJentzen.h2_o[index_t](grad)   # grad is a matrix of shape (batch_size, dim) 
        
        #pde = f + 0.5*sigma**2*tr_hess + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1)
        pde = f + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1) # we just take the part of the pde where alpha is involved
        loss = pde

        # backwards step
        loss.backward()
        
        # optimizer step
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.cpu().data[0][0]))
        
    #return model
    
def policy_improvement_step_new(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep, law):
    
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #model = Net_alpha(dim)
    netAlpha.train()
    netJentzen.eval()
    
    optimizer = torch.optim.Adam(netAlpha.parameters(),lr=base_lr)
    #optimizer = torch.optim.SGD(netAlpha.parameters(),lr=base_lr, momentum=0.9, nesterov=True)
    
    for it in range(n_iter):
        # optimisation step decay
        lr = base_lr * (0.5 ** (it // 1000))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        
        optimizer.zero_grad()        
        if cuda:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim])).cuda()
        else:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim]))
        x = Variable(x)
        #index_t = random.randint(0,len(timegrid)-2)
        #t = torch.ones(1,1)*timegrid[index_t]
        t_numpy = np.random.choice(timegrid[:-1], size=(batch_size, 1))
        if cuda:
            t = Variable(torch.Tensor(t_numpy).cuda())
        else:
            t = Variable(torch.Tensor(t_numpy))
        tx = torch.cat([t,x], 1)
        alpha_tx = netAlpha(tx)
        grad = netJentzen.i_h1(tx)
        grad = netJentzen.h1_h2(grad)
        grad = netJentzen.h2_o(grad)
        
        
        #pde = f + 0.5*sigma**2*tr_hess + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1)
        # we define H(t,x,alpha,grad_v) = f(t,x) + b(t,x,alpha)*grad_v(t,x). This is what we want ot minimise in terms of alpha. Therefore we want grad_alpha H to be 0
        grad_H = alpha_tx + grad
        
        #pde = f + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1) # we just take the part of the pde where alpha is involved
        loss = torch.sum(torch.norm(grad_H,2,1)**2)

        # backwards step
        loss.backward()
        
        # optimizer step
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.cpu().data[0]))
        
    #return model
        

        
def law_improvement_step(netJentzen, init_values):
    if cuda:
        netJentzen.cuda()
    netJentzen.eval()
    improved_law = []
    for player in range(init_values.size()[0]):
        law_player = []
        print('MonteCarlo player {}/{}'.format(player, init_values.size()[0]))
        if cuda:
            input = Variable(torch.cat([init_values[player].data.view(1,-1)]*n_iter, dim=0).cuda())
        else:
            input = Variable(torch.cat([init_values[player].data.view(1,-1)]*n_iter, dim=0))
        _,_,path = netJentzen(input)
        for step in path:
            law_player.append(step.mean(0).view(1,-1))
        law_player = torch.cat(law_player, 0)
        improved_law.append(law_player)
    law = sum(improved_law)/len(improved_law)
    return law
    

    
def main():
    
    batch_size=500
    base_lr=0.05
    n_iter=200
    dim = 100
    kappa = 1
    sigma = 0.1
    init_t = 0
    T = 2
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    
    # number of players and init_values
    n_players = 20
    if cuda:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim])).cuda()
    else:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim]))
    init_values = Variable(init_values)
    
       
    # 0. We initialise the law
    if cuda:
        law = Variable((torch.zeros([timegrid.size, dim])).cuda())
    else:
        law     = Variable(torch.zeros([timegrid.size, dim]))
    
        
    # 1. we initialise alpha
    netAlpha = Net_alpha(dim=dim)
    if cuda:
        netAlpha.cuda()
    
    # 2. evaluation step: approximation of v to PDE    
    netJentzen = Net_Jentzen_1network(dim=dim, kappa=kappa, sigma=sigma, law=law, timegrid=timegrid)
    if cuda:
        netJentzen.cuda()
    netJentzen.train()
    netAlpha.eval()    
    evaluation_step(batch_size=batch_size, base_lr=base_lr, 
                    n_iter=500, dim=dim, kappa=kappa, 
                    sigma=sigma, init_t=init_t, T=T, timestep=timestep, 
                    law=law)
    # 3. Policy optimisation step: we minise alpha
    netAlpha = Net_alpha(dim=dim)
    if cuda:
        netAlpha.cuda()
    netAlpha.train()
    netJentzen.eval()
    policy_improvement_step_new(batch_size=3000, base_lr=0.02, n_iter=1000, 
                            dim=dim, kappa=kappa, sigma=sigma, 
                            init_t=init_t, T=T, timestep=timestep, 
                            law=law)
    
    # we have to repeat the above until convergence
    
    # 4. Law improvement: Monte Carlo to improve the law
    netJentzen.eval()
    law = law_improvement_step(netJentzen, init_values)
    

        
        
        
    
#################### TEST
netJentzen.eval()
netAlpha.eval()
l = []
for i in range(init_values.size()[0]):
    x = init_values[i].view(1,-1)
    v,_,_ = netJentzen(x)
    l.append(v)




player = 0
x = init_values[player].view(1,-1)
v0 = netJentzen.v0_i_h1(x)
v0 = netJentzen.v0_h1_h2(v0)
v0 = netJentzen.v0_h2_o(v0)

t = (torch.ones(1,1)*timegrid[23]).cuda()
t = Variable(t)
tx = torch.cat([t,x], 1)
grad = netJentzen.i_h1(tx)
grad = netJentzen.h1_h2(grad)
grad = netJentzen.h2_o(grad)
alpha = netAlpha(tx)

paths = []
netJentzen.eval()
for player in range(init_values.size()[0]):
        print('player {}'.format(player))
        x = init_values[player].view(1,-1)
        v,_,path = netJentzen(x)
        path = torch.cat(path, 0)
        paths.append(path.cpu().data.numpy())
    
    
os.chdir('/floydhub')
for i in range(len(paths)):
    np.savetxt('path_'+str(i)+'.txt', paths[i])
    
path = glob.glob('path*')
    
paths = []
for f in path:  
    paths.append(np.loadtxt(f))
    

fig, ax = plt.subplots(figsize=(8,5))
for player in paths:
    ax.plot(player[:,0])  # we plot the first coordinate

    

    
###################################################################    
########### TEST CODE TO TEST CALCULATION OF HESSIAN ##############
###################################################################
m = nn.Linear(10, 10)
x = Variable(torch.randn(20,10), requires_grad=True)
grad = m(x)
diag_hess = []
for col in range(x.size()[1]):  # each col is a dimension
    l = [grad[row][col] for row in range(grad.size()[0])]
    hess = torch.autograd.grad(l, x, create_graph=True)[0]
    diag_hess.append(hess[:,col].contiguous().view(-1,1))
diag_hess = torch.cat(diag_hess, 1)
tr_hess = diag_hess.sum(1)

x.requires_grad=False
grad = m(x)
h = 0.001
diag_hess = []
for col in range(x.size()[1]):
    y = copy.deepcopy(x)
    y[:,col] = y[:,col]+h
    grad_h = m(y)
    hess =(grad_h-grad)/h
    diag_hess.append(hess[:,col].contiguous().view(-1,1))
diag_hess_diff = torch.cat(diag_hess,1)
tr_hess_diff = diag_hess_diff.sum(1)
return tr_hess

    