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


def save_checkpoint(state, it, value=True):
    if value:
        filename = '/floydhub/model_value_'+str(it)+'.pth.tar'
    else:
        filename = '/floydhub/model_alpha_'+str(it)+'.pth.tar'
    torch.save(state, filename)


def save_law(law, it):
    law_numpy = law.data.cpu().numpy()
    np.savetxt('/floydhub/law_'+str(it)+'.txt', law_numpy)



class Net_alpha(nn.Module):
    """
    We create the alpha net
    """
    
    def __init__(self, dim):
        super(Net_alpha, self).__init__()
        self.dim = dim
        
        self.i_h1 = self.hiddenLayer(dim+1, dim+20)  # dim+1 because we have time
        self.h1_h2 = self.hiddenLayer(dim+20, dim+20)
        self.h2_h3 = self.hiddenLayer(dim+20, dim+20)
        self.h3_h4 = self.hiddenLayer(dim+20, dim+20)
        self.h4_h5 = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6 = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6 = self.hiddenLayer(dim+20, dim+20)
        self.h6_h7 = self.hiddenLayer(dim+20, dim+20)
        self.h7_h8 = self.hiddenLayer(dim+20, dim+20)
        self.h8_h9 = self.hiddenLayer(dim+20, dim+20)
        self.h9_h10 = self.hiddenLayer(dim+20, dim+20)
        self.h10_h11 = self.hiddenLayer(dim+20, dim+20)
        self.h11_h12 = self.hiddenLayer(dim+20, dim+20)
        self.h12_h13 = self.hiddenLayer(dim+20, dim+20)
        self.h13_h14 = self.hiddenLayer(dim+20, dim+20)
        self.h14_h15 = self.hiddenLayer(dim+20, dim+20)
        self.h15_o = nn.Linear(dim+20, dim) 
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut),
                              nn.BatchNorm1d(nOut),
                              nn.ReLU())
        return layer
    
    def forward(self, tx):
        h1 = self.i_h1(tx)
        h2 = self.h1_h2(h1)
        h3 = self.h2_h3(h2)
        h4 = self.h3_h4(h3)
        h5 = self.h4_h5(h4)
        h6 = self.h5_h6(h5)
        h7 = self.h6_h7(h6)
        h8 = self.h7_h8(h7)
        h9 = self.h8_h9(h8)
        h10 = self.h9_h10(h9)
        h11 = self.h10_h11(h10)
        h12 = self.h11_h12(h11)
        h13 = self.h12_h13(h12)
        h14 = self.h13_h14(h13)
        h15 = self.h14_h15(h14)
        alpha = self.h15_o(h15)
        return alpha



class Net_Jentzen_1network(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, kappa, sigma, timegrid):
        super(Net_Jentzen_1network, self).__init__()
        self.dim = dim
        if cuda:
            self.timegrid = torch.Tensor(timegrid).cuda()
        else:
            self.timegrid = torch.Tensor(timegrid)
        #self.law = law # law is a matrix with the same number of rows as timegrid (one per each timestep), and the same number of columns as dim
        self.kappa = kappa
        self.sigma = sigma
        netAlpha.eval()
        
        self.i_h1 = self.hiddenLayer(dim+1, dim+20)  # dim+1 because we have time
        self.h1_h2 = self.hiddenLayer(dim+20, dim+20)
        self.h2_h3 = self.hiddenLayer(dim+20, dim+20)
        self.h3_h4 = self.hiddenLayer(dim+20, dim+20)
        self.h4_h5 = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6 = self.hiddenLayer(dim+20, dim+20)
        self.h6_h7 = self.hiddenLayer(dim+20, dim+20)
        self.h7_h8 = self.hiddenLayer(dim+20, dim+20)
        self.h8_h9 = self.hiddenLayer(dim+20, dim+20)
        self.h9_h10 = self.hiddenLayer(dim+20, dim+20)
        self.h10_h11 = self.hiddenLayer(dim+20, dim+20)
        self.h11_h12 = self.hiddenLayer(dim+20, dim+20)
        self.h12_h13 = self.hiddenLayer(dim+20, dim+20)
        self.h13_h14 = self.hiddenLayer(dim+20, dim+20)
        self.h14_h15 = self.hiddenLayer(dim+20, dim+20)
        self.h15_o = nn.Linear(dim+20, dim)  
                
        self.v0_i_h1 = self.hiddenLayer(dim, dim+20)
        self.v0_h1_h2 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h2_h3 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h3_h4 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h4_h5 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h5_h6 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h6_h7 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h7_h8 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h8_h9 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h9_h10 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h10_h11 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h11_h12 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h12_h13 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h13_h14 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h14_h15 = self.hiddenLayer(dim+20, dim+20)
        self.v0_h15_o = nn.Linear(dim+20, 1)
        
    
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
            h3 = self.h2_h3(h2)
            h4 = self.h3_h4(h3)
            h5 = self.h4_h5(h4)
            h6 = self.h5_h6(h5)
            h7 = self.h6_h7(h6)
            h8 = self.h7_h8(h7)
            h9 = self.h8_h9(h8)
            h10 = self.h9_h10(h9)
            h11 = self.h10_h11(h10)
            h12 = self.h11_h12(h11)
            h13 = self.h12_h13(h12)
            h14 = self.h13_h14(h13)
            h15 = self.h14_h15(h14)
            grad = self.h15_o(h15)
            
            #kernel1 = torch.exp(-(x-self.law[i]-(-0.8))**2/(2*0.08**2))
            #kernel2 = torch.exp(-(x-self.law[i]-(0.8))**2/(2*0.08**2))
            #kernel = kernel1+kernel2
            #f = torch.norm(kernel,2,1)**2 +  0.5*torch.norm(alpha,2,1)**2

            
            f = (self.kappa)/2 * torch.norm(x-law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
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
                v0 = self.v0_h2_h3(v0)
                v0 = self.v0_h3_h4(v0)
                v0 = self.v0_h4_h5(v0)
                v0 = self.v0_h5_h6(v0)
                v0 = self.v0_h6_h7(v0)
                v0 = self.v0_h7_h8(v0)
                v0 = self.v0_h8_h9(v0)
                v0 = self.v0_h9_h10(v0)
                v0 = self.v0_h10_h11(v0)
                v0 = self.v0_h11_h12(v0)
                v0 = self.v0_h12_h13(v0)
                v0 = self.v0_h13_h14(v0)
                v0 = self.v0_h14_h15(v0)
                v0 = self.v0_h15_o(v0)
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
    
    def __init__(self, dim, kappa, sigma, timegrid):
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
            
            f = (self.kappa)/2 * torch.norm(x-law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
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
    
    
    
def evaluation_step(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep, eps):
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
    #for it in range(n_iter):
    value_converges = False
    while (not value_converges) and it<400:
        #while it<n_iter:
        it +=1
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
        if loss.data.cpu()[0]<eps:
            value_converges = True
        

    #return model
    
    


    
def policy_improvement_step_new(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep, eps):
    
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #model = Net_alpha(dim)
    netAlpha.train()
    netJentzen.eval()
    
    optimizer = torch.optim.Adam(netAlpha.parameters(),lr=base_lr)
    #optimizer = torch.optim.SGD(netAlpha.parameters(),lr=base_lr, momentum=0.9, nesterov=True)
    
    alpha_converges = False
    #for it in range(n_iter):
    it = 0
    while (not alpha_converges) and it<1000:
        #while it<n_iter:
        # optimisation step decay
        it+=1
        lr = base_lr * (0.5 ** (it // 500))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        
        optimizer.zero_grad()        
        if cuda:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim])).cuda()
        else:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim]))
        x = Variable(x)
        t_numpy = np.random.choice(timegrid[:-1], size=(batch_size, 1))
        if cuda:
            t = Variable(torch.Tensor(t_numpy).cuda())
        else:
            t = Variable(torch.Tensor(t_numpy))
        tx = torch.cat([t,x], 1)
        alpha_tx = netAlpha(tx)
        grad = netJentzen.i_h1(tx)
        grad = netJentzen.h1_h2(grad)
        grad = netJentzen.h2_h3(grad)
        grad = netJentzen.h3_h4(grad)
        grad = netJentzen.h4_h5(grad)
        grad = netJentzen.h5_h6(grad)
        grad = netJentzen.h6_h7(grad)
        grad = netJentzen.h7_h8(grad)
        grad = netJentzen.h8_h9(grad)
        grad = netJentzen.h9_h10(grad)
        grad = netJentzen.h10_h11(grad)
        grad = netJentzen.h11_h12(grad)
        grad = netJentzen.h12_h13(grad)
        grad = netJentzen.h13_h14(grad)
        grad = netJentzen.h14_h15(grad)
        grad = netJentzen.h15_o(grad)

        # we define H(t,x,alpha,grad_v) = f(t,x) + b(t,x,alpha)*grad_v(t,x). This is what we want ot minimise in terms of alpha. Therefore we want grad_alpha H to be 0
        grad_H = alpha_tx + grad
        
        #pde = f + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1) # we just take the part of the pde where alpha is involved
        loss = 1/batch_size*torch.sum(torch.norm(grad_H,2,1)**2)

        # backwards step
        loss.backward()
        if loss.cpu().data[0]<eps:
            alpha_converges = True
        
        # optimizer step
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.cpu().data[0]))
        
    #return model
    

def law_improvement_step(init_values, n_iter = 1000):
    """
    Monte Carlo based law improvement step
    """
    if cuda:
        netJentzen.cuda()
    netJentzen.eval()
    netAlpha.eval()
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
    law = Variable(law.data)
    return law    

    
def evaluation_improvement_law_new(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep):
    batch_size=1000
    base_lr=0.005
    n_iter=50
    dim = 10
    kappa = 10
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    
    # number of players and init_values
    n_players = 20
    if cuda:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim])).cuda()
        batch_space_to_measure_alpha = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1000, dim])).cuda()
    else:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim]))
        batch_space_to_measure_alpha = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1000, dim]))

    batch_space_to_measure_alpha = Variable(batch_space_to_measure_alpha)
    init_values = Variable(init_values)
    
    # 0. We initialise the law
    if cuda:
        law = Variable((torch.zeros([timegrid.size, dim])).cuda())
    else:
        law = Variable(torch.zeros([timegrid.size, dim]))
    
        
    # 1. we initialise alpha
    netAlpha = Net_alpha(dim=dim)
    if cuda:
        netAlpha.cuda()
    
    # 2. we initialise the value function
    netJentzen = Net_Jentzen_1network(dim=dim, kappa=kappa, sigma=sigma, timegrid=timegrid)
    if cuda:
        netJentzen.cuda()

    timegrid = np.arange(init_t, T+timestep/2, timestep)
    if cuda:
        netJentzen.cuda()
    if cuda:
        netAlpha.cuda()
        
    
    

    it = 0
    laws = [law]    
    alphas =[getNorm_alpha_from_batch(batch_space_to_measure_alpha)]
    norm_values = [getNorm_value_from_batch(batch_space_to_measure_alpha)]
    for it in range(n_iter):
        print('Iteration {}/{}'.format(it, n_iter))
        #######################################
        # Gradient Descent for value function #
        #######################################
        evaluation_step(batch_size=batch_size, base_lr=base_lr, 
                        dim=dim, kappa=kappa, sigma=sigma, init_t=init_t, 
                        T=T, timestep=timestep, n_iter=30, eps=0.1)
        a = getNorm_value_from_batch(batch_space_to_measure_alpha)
        norm_values.append(a)
        
        # Gradient descent for policy function
        policy_improvement_step_new(batch_size=batch_size, base_lr=base_lr, 
                                    dim=dim, kappa=kappa, sigma=sigma, 
                                    init_t=init_t, T=T, timestep=timestep, n_iter=30, eps=0.1)
        alphas.append(getNorm_alpha_from_batch(batch_space_to_measure_alpha))
        
        # Monte Carlo based law improvement
        if (it+1)%10==0: 
            law = law_improvement_step(init_values)
            laws.append(law)
            





    
    
#    netAlpha.eval()
#    x1 = np.arange(-1,1,0.01)
#    xx = []
#    policies = []
#    for xx1 in x1:
#        coord1 = torch.Tensor([xx1]).cuda().view(1,-1)
#        x = torch.cat([coord1,torch.zeros(1, dim-1).cuda()],1)
#        x = Variable(x)
#        xx.append(x)
#        pol = torch.zeros([len(timegrid)-1, dim])
#        for i in range(len(timegrid)-1):
#            t = Variable((torch.ones(1,1)*timegrid[i]).cuda())            
#            tx = torch.cat([t,x], dim=1)
#            alpha_tx = netAlpha(tx)
#            pol[i] = alpha_tx.data.view(-1)
#
#        policies.append(pol)
#        
#    pol_1st = [p[:,0].contiguous().view(-1,1) for p in policies]  # each row: time. Each column: x
#    pol_1st = torch.cat(pol_1st, 1).numpy()  # matrix of shape timegrid x xgrid
#    np.savetxt('pol_1st.txt', pol_1st)
#    pol_1st = np.loadtxt('pol_1st.txt')
#        
#    X_grid, Y_grid = np.meshgrid(x1, timegrid[:-1])
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    ax.plot_surface(X_grid, Y_grid, pol_1st, cmap="coolwarm",
#                           linewidth=0, antialiased=False)

        
        
    
def getNorm_alpha_from_batch(batch_space_to_measure_alpha):
    """
    Since we have exponential number of points in a grid of dimension 100,
    we will get the norm of the alpha at timegrid x init_values
    """
    netAlpha.eval()
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    input = batch_space_to_measure_alpha
    if cuda:
        t = torch.cat([torch.ones((batch_space_to_measure_alpha.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
        t = t.cuda()
    else:
        t = torch.cat([torch.ones((batch_space_to_measure_alpha.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
    
    t = Variable(t)
    input = torch.cat([input]*(len(timegrid)-1),0)
    tx = torch.cat([t,input],1)
    alpha_tx = netAlpha(tx)
    return alpha_tx

def getNorm_value_from_batch(batch_space_to_measure_alpha):
    """
    Since we have exponential number of points in a grid of dimension 100,
    we will get the norm of the alpha at timegrid x init_values
    """
    netJentzen.eval()
    
    x = batch_space_to_measure_alpha
    
    v0 = netJentzen.v0_i_h1(x)
    v0 = netJentzen.v0_h1_h2(v0)
    v0 = netJentzen.v0_h2_h3(v0)
    v0 = netJentzen.v0_h3_h4(v0)
    v0 = netJentzen.v0_h4_h5(v0)
    v0 = netJentzen.v0_h5_h6(v0)
    v0 = netJentzen.v0_h6_h7(v0)
    v0 = netJentzen.v0_h7_h8(v0)
    v0 = netJentzen.v0_h8_h9(v0)
    v0 = netJentzen.v0_h9_h10(v0)
    v0 = netJentzen.v0_h10_h11(v0)
    v0 = netJentzen.v0_h11_h12(v0)
    v0 = netJentzen.v0_h12_h13(v0)
    v0 = netJentzen.v0_h13_h14(v0)
    v0 = netJentzen.v0_h14_h15(v0)
    v0 = netJentzen.v0_h15_o(v0)
    
    return v0
    
    
        
    
def algorithm1():
    
    batch_size=1000
    base_lr=0.005
    n_iter=200
    dim = 10
    kappa = 5
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    eps = 20
    
    # number of players and init_values
    n_players = 30
    if cuda:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim])).cuda()
        batch_space_to_measure_alpha = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1000, dim])).cuda()
    else:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim]))
        batch_space_to_measure_alpha = torch.Tensor(np.random.uniform(low=-1, high=1, size=[1000, dim]))

    init_values = Variable(init_values)
    batch_space_to_measure_alpha = Variable(batch_space_to_measure_alpha)    
       
    # 0. We initialise the law
    if cuda:
        law = Variable((torch.zeros([timegrid.size, dim])).cuda())
    else:
        law = Variable(torch.zeros([timegrid.size, dim]))
    laws = [law]
        
    
    law_ierations = 5
    max_law_it = 5

    netAlpha = Net_alpha(dim=dim)
    if cuda:
        netAlpha.cuda()

    netJentzen = Net_Jentzen_1network(dim=dim, kappa=kappa, sigma=sigma, timegrid=timegrid)
    if cuda:
        netJentzen.cuda()
    

    norm_alphas = []
    norm_values = []
    for law_it in range(law_ierations):
        
        #while not alpha_converges:
        #for i in range(3):
        alpha_converges = False
        norm_alphas_iteration = []
        netAlpha.eval()
        a = getNorm_alpha_from_batch(batch_space_to_measure_alpha)
        norm_alphas_iteration.append(a)
        max_law_it = 0
        while not alpha_converges:
            max_law_it += 1
            # 2. evaluation step: approximation of v to PDE              
            netJentzen.train()
            netAlpha.eval()   
            evaluation_step(batch_size=1000, base_lr=0.005, 
                            n_iter=300, dim=dim, kappa=kappa, 
                            sigma=sigma, init_t=init_t, T=T, timestep=timestep, eps=0.6) 
            save_checkpoint(netJentzen.state_dict(), law_it, value=True)
            a = getNorm_value_from_batch(batch_space_to_measure_alpha)
            norm_values.append(a)
            # 3. Policy optimisation step: we minise alpha
            netAlpha.train()
            netJentzen.eval()
            policy_improvement_step_new(batch_size=4000, base_lr=0.005, n_iter=1000, #n_iter = 3000, 
                                    dim=dim, kappa=kappa, sigma=sigma, 
                                    init_t=init_t, T=T, timestep=timestep, eps=0.4)
            save_checkpoint(netAlpha.state_dict(), law_it, value=False)
                                    
            netAlpha.eval()
            a = getNorm_alpha_from_batch(batch_space_to_measure_alpha)
            norm_alphas_iteration.append(a)
            norm_alphas.append(a)
            
            
            # we check for convergence
            if len(norm_alphas_iteration)>1:
                diff = norm_alphas_iteration[-1]-norm_alphas_iteration[-2]
                n = torch.sum(torch.norm(diff,2,1)**2)/norm_alphas_iteration[-1].size()[0]
                n = n.data.cpu()[0]
                print("norm difference alphas is {:.4f}".format(n))
                if n<1 or max_law_it>5:
                    alpha_converges=True
    
        
        # 4. Law improvement: Monte Carlo to improve the law
        netJentzen.eval()
        netAlpha.eval()
        law = law_improvement_step(init_values)
        if cuda:
            law = law.cpu().data
            law = Variable(law.cuda())
        else:
            law = law.data
            law = Variable(law)
        laws.append(law)
        save_law(law, law_it)


# compare laws
norms = []
for i in range(len(laws)-1):
    norms.append(torch.sum(torch.norm(laws[i+1]-laws[i],2,1)**2)/laws[i].size()[0])
norms = [norms[i].data.cpu()[0] for i in range(len(norms))]
normsLaw = np.array(norms)
np.savetxt('normsLaw.txt', normsLaw)

normsA = []
for i in range(len(alphas)-1):
    normsA.append(torch.sum(torch.norm(alphas[i+1]-alphas[i],2,1)**2)/alphas[i].size()[0])
normsA = [normsA[i].data.cpu()[0] for i in range(len(normsA))]
norsmA = np.array(normsA)
np.savetxt('normsAlpha.txt', normsA)

os.chdir('/Users/msabate/Projects/Turing/Mean-Field-Games/src/NIPS18/images')
normsA = np.loadtxt('normsAlpha.txt')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(normsA, 'o-')


os.chdir('/Users/msabate/Projects/Turing/Mean-Field-Games/src/NIPS18/images')
normsLaw = np.loadtxt('normsLaw.txt')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(normsLaw, 'o-')


    
#################### TEST
netJentzen.eval()
netAlpha.eval()
l = []
for i in range(init_values.size()[0]):
    x = init_values[i].view(1,-1)
    v,_,_ = netJentzen(x)
    l.append(v)




player = 1
x = init_values[player].view(1,-1)
v0 = netJentzen.v0_i_h1(x)
v0 = netJentzen.v0_h1_h2(v0)
v0 = netJentzen.v0_h2_o(v0)

t = (torch.ones(1,1)*timegrid[0]).cuda()
t = Variable(t)
tx = torch.cat([t,x], 1)
grad = netJentzen.i_h1(tx)
grad = netJentzen.h1_h2(grad)
grad = netJentzen.h2_h3(grad)
grad = netJentzen.h3_o(grad)
alpha = netAlpha(tx)

paths = []
netJentzen.eval()
netAlpha.eval()
for player in range(init_values.size()[0]):
    print('player {}'.format(player))
    x = init_values[player].view(1,-1)
    v,_,path = netJentzen(x)
    path = torch.cat(path, 0)
    paths.append(path.cpu().data.numpy())
    
    
os.chdir('/floydhub')
for i in range(len(paths)):
    np.savetxt('path_'+str(i)+'.txt', paths[i])


import glob
import numpy as np
import matplotlib.pyplot as plt    
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

    