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
from collections import namedtuple

cuda = True


class Net_Algorithm2(nn.Module):
    """
    Szpruch and Saski's network
    """
    
    def __init__(self, dim):
        super(Net_Algorithm2, self).__init__()
        self.dim = dim
        
        self.i_h1_grad =  self.hiddenLayer(dim+1, dim+20)   # dim+1 because the input to the network is (t,x)
        self.h1_h2_grad = self.hiddenLayer(dim+20, dim+20)
        #self.h2_o_grad = nn.Linear(dim+20, dim)
        self.h2_h3_grad = self.hiddenLayer(dim+20, dim+20)
        self.h3_h4_grad = self.hiddenLayer(dim+20, dim+20)
        self.h4_h5_grad = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6_grad = self.hiddenLayer(dim+20, dim+20)
        self.h6_h7_grad = self.hiddenLayer(dim+20, dim+20)
        self.h7_h8_grad = self.hiddenLayer(dim+20, dim+20)
        self.h8_h9_grad = self.hiddenLayer(dim+20, dim+20)
        self.h9_h10_grad = self.hiddenLayer(dim+20, dim+20)
        self.h10_o_grad = nn.Linear(dim+20, dim)
        
        self.i_h1_v =  self.hiddenLayer(dim+1, dim+20)   # dim+1 because the input to the network is (t,x)
        self.h1_h2_v = self.hiddenLayer(dim+20, dim+20)
        #self.h2_o_v = nn.Linear(dim+20, 1)
        self.h2_h3_v = self.hiddenLayer(dim+20, dim+20)
        self.h3_h4_v = self.hiddenLayer(dim+20, dim+20)
        self.h4_h5_v = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6_v = self.hiddenLayer(dim+20, dim+20)
        self.h6_h7_v = self.hiddenLayer(dim+20, dim+20)
        self.h7_h8_v = self.hiddenLayer(dim+20, dim+20)
        self.h8_h9_v = self.hiddenLayer(dim+20, dim+20)
        self.h9_h10_v = self.hiddenLayer(dim+20, dim+20)
        self.h10_o_v = nn.Linear(dim+20, 1)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut),
                              nn.BatchNorm1d(nOut),
                              nn.ReLU())
        return layer
        
    def forward(self, x):
        # first coordinate of x is time
        h1_grad = self.i_h1_grad(x)
        h2_grad = self.h1_h2_grad(h1_grad)
        h3_grad = self.h2_h3_grad(h2_grad)
        h4_grad = self.h3_h4_grad(h3_grad)
        h5_grad = self.h4_h5_grad(h4_grad)
        h6_grad = self.h5_h6_grad(h5_grad)
        h7_grad = self.h6_h7_grad(h6_grad)
        h8_grad = self.h7_h8_grad(h7_grad)
        h9_grad = self.h8_h9_grad(h8_grad)
        h10_grad = self.h9_h10_grad(h9_grad)
        grad = self.h10_o_grad(h10_grad) 
        
        h1_v = self.i_h1_v(x)
        h2_v = self.h1_h2_v(h1_v)
        h3_v = self.h2_h3_v(h2_v)
        h4_v = self.h3_h4_v(h3_v)
        h5_v = self.h4_h5_v(h4_v)
        h6_v = self.h5_h6_v(h5_v)
        h7_v = self.h6_h7_v(h6_v)
        h8_v = self.h7_h8_v(h7_v)
        h9_v = self.h8_h9_v(h8_v)
        h10_v = self.h9_h10_v(h9_v)
        v = self.h10_o_v(h10_v)
        
        return v, grad
    
    

    

class Net_alpha(nn.Module):
    """
    We create the alpha net
    """
    
    def __init__(self, dim):
        super(Net_alpha, self).__init__()
        self.dim = dim
        
        self.i_h1 = self.hiddenLayer(dim+1, dim+20)
        self.h1_h2 = self.hiddenLayer(dim+20, dim+20)
        self.h2_h3 = self.hiddenLayer(dim+20, dim+20)
        self.h3_h4 = self.hiddenLayer(dim+20, dim+20)
        self.h4_h5 = self.hiddenLayer(dim+20, dim+20)
        self.h5_h6 = self.hiddenLayer(dim+20, dim+20)
        self.h6_h7 = self.hiddenLayer(dim+20, dim+20)
        self.h7_h8 = self.hiddenLayer(dim+20, dim+20)
        self.h8_h9 = self.hiddenLayer(dim+20, dim+20)
        self.h9_h10 = self.hiddenLayer(dim+20, dim+20)
        self.h10_o = nn.Linear(dim+20, dim)
        
    
    def hiddenLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut),
                              #nn.BatchNorm1d(nOut),
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
        alpha = self.h10_o(h10)
        return alpha


def save_checkpoint(state, it, value=True):
    if value:
        filename = '/floydhub/model_value_'+str(it)+'.pth.tar'
    torch.save(state, filename)


def save_law(law, it):
    law_numpy = law.data.cpu().numpy()
    np.savetxt('/floydhub/law_'+str(it)+'.txt', law_numpy)
    
        
def evaluation_step(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep):
    """
    This function solves the PDE, with fixed law and alpha (control)
    Inputs:
        - kappa, sigma: floats
        - timegrid: 
    """
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    if cuda:
        netValue.cuda()
    
    optimizer = torch.optim.Adam(netValue.parameters(),lr=base_lr)
    
    netValue.train()
    netAlpha.eval()
    
    output_model = namedtuple('output', ['v', 'grad_v'])
    
    it = 0
    for it in range(n_iter):
        
        # parameter decay
        #lr = base_lr * (0.5 ** (it // 50))
        #for param_group in optimizer.state_dict()['param_groups']:
        #    param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        if cuda:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim])).cuda()
        else:
            x = torch.Tensor(np.random.uniform(low=-1, high=1, size=[batch_size, dim]))
        x = Variable(x)
        if cuda:
            t = torch.zeros(batch_size, 1).cuda()
        else:
            t = torch.zeros(batch_size, 1)
        t = Variable(t)
        tx = torch.cat([t,x],1)
        #alpha_tx = netAlpha(tx)
        v, grad_v  = netValue(tx) 
        
        brownian = [output_model(v, grad_v)]  # we will save the brownian path
        error = []
        
        for i in range(1,len(timegrid)):
            alpha_tx = netAlpha(tx)
            h = timegrid[i]-timegrid[i-1]
            if cuda:
                #t = (torch.ones(batch_size,1)*timegrid[i]).cuda()
                xi = Variable(torch.randn(batch_size, dim).cuda())
            else:
                #t = torch.ones(batch_size,1)*timegrid[i]
                xi = Variable(torch.randn(batch_size, dim))
            
            
            f = (kappa)/2 * torch.norm(x-law[i],2,1)**2 +  0.5*torch.norm(alpha_tx,2,1)**2
            f = f.view(-1,1)
            
            if cuda:
                x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi
            else:
                x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi # to complete - make it general for any LQR problem
            
            if cuda:
                t = (torch.ones(batch_size,1)*timegrid[i]).cuda()
            else:
                t = torch.ones(batch_size,1)*timegrid[i]
            t = Variable(t)
            tx = torch.cat([t,x], dim=1)
            v, grad_v  = netValue(tx) 
            
            brownian.append(output_model(v, grad_v))
            if cuda:
                error.append(brownian[-1].v - brownian[-2].v + 
                             f*h - 
                             torch.diag(torch.matmul(brownian[-2].grad_v, sigma*math.sqrt(h)*xi.transpose(1,0))))
            else:
                error.append(brownian[-1].v - brownian[-2].v + 
                             f*h - 
                             torch.diag(torch.matmul(brownian[-2].grad_v, sigma*math.sqrt(h)*xi.transpose(1,0))))


            
        # we build the loss function from Raissi's paper
        error = torch.cat(error, dim=1)
        if cuda:
            target_terminal = Variable(torch.zeros(batch_size, 1).cuda())
        else:
            target_terminal = Variable(torch.zeros(batch_size, 1))
        
        error_terminal = brownian[-1].v - target_terminal
        loss = torch.sum(torch.norm(error,2,1)**2) + torch.sum(torch.norm(error_terminal,2,1)**2)
        loss.backward()
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
        
    print('Its over!!!')
    
    
def policy_improvement_step_new(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep):
    
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #model = Net_alpha(dim)
    netAlpha.train()
    netValue.eval()
    
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
        t_numpy = np.random.choice(timegrid[:-1], size=(batch_size, 1))
        if cuda:
            t = Variable(torch.Tensor(t_numpy).cuda())
        else:
            t = Variable(torch.Tensor(t_numpy))
        tx = torch.cat([t,x], 1)
        alpha_tx = netAlpha(tx)
        _, grad_tx = netValue(tx)
                
        # we define H(t,x,alpha,grad_v) = f(t,x) + b(t,x,alpha)*grad_v(t,x). This is what we want ot minimise in terms of alpha. Therefore we want grad_alpha H to be 0
        grad_H = alpha_tx + grad_tx
        
        #pde = f + torch.diag(torch.matmul(alpha_tx, grad.transpose(1,0))).view(-1,1) # we just take the part of the pde where alpha is involved
        loss = torch.sum(torch.norm(grad_H,2,1)**2)

        # backwards step
        loss.backward()
        
        # optimizer step
        optimizer.step()
        print("Iteration=[{it}/{n_iter}]\t loss={loss:.5f}".format(it=it, n_iter=n_iter, loss=loss.cpu().data[0]))
        
        
        
        
        
        

def get_path(x, n_iter, dim, sigma, init_t, T, timestep):
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    netAlpha.eval()
    x = torch.cat([x.view(1,-1)]*n_iter, 0)
    path = [Variable(x.data)]
    
    for i in range(len(timegrid)-1):
        h = timegrid[i+1]-timegrid[i]
        if cuda:
            t = (torch.ones(n_iter,1)*timegrid[i]).cuda()
            xi = Variable(torch.randn(n_iter, dim).cuda())
        else:
            t = torch.ones(n_iter,1)*timegrid[i]
            xi = Variable(torch.randn(n_iter, dim))
            
        t = Variable(t)
        tx = torch.cat([t,x],1)
        alpha_tx = netAlpha(tx)            
        if cuda:
            x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi
        else:
            x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi # to complete - make it general for any LQR problem
        path.append(Variable(x.data))
    return path # list of length len(timegrid). Each element of the list is a timestep, a matrix of size (n_iter, dim)



def law_improvement_step(n_iter, dim, sigma, init_t, T, timestep):
    """
    Monte Carlo based law improvement step
    """
    if cuda:
        netValue.cuda()
    netValue.eval()
    netAlpha.eval()
    improved_law = []
    for player in range(init_values.size()[0]):
        law_player = []
        print('MonteCarlo player {}/{}'.format(player, init_values.size()[0]))
        if cuda:
            input = Variable(torch.cat([init_values[player].data.view(1,-1)]*n_iter, dim=0).cuda())
        else:
            input = Variable(torch.cat([init_values[player].data.view(1,-1)]*n_iter, dim=0))
        path = get_path(input, n_iter, dim, sigma, init_t, T, timestep)
        for step in path:
            law_player.append(step.mean(0).view(1,-1))
        law_player = torch.cat(law_player, 0)
        improved_law.append(law_player)
    law = sum(improved_law)/len(improved_law)
    law = Variable(law.data)
    return law    



def getNorm_alpha_from_batch(batch_space_to_measure_alpha):
    """
    Since we have exponential number of points in a grid of dimension 100,
    we will get the norm of the alpha at timegrid x init_values
    """
    netAlpha.eval()
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    input = batch_space_to_measure_alpha
    if cuda:
        t = torch.cat([torch.ones((input.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
        t = t.cuda()
    else:
        t = torch.cat([torch.ones((input.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
    
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
    netAlpha.eval()
    netValue.eval()
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    input = batch_space_to_measure_alpha
    if cuda:
        t = torch.cat([torch.ones((input.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
        t = t.cuda()
    else:
        t = torch.cat([torch.ones((input.size()[0], 1))*tt for tt in timegrid[:-1]], 0)
    
    t = Variable(t)
    input = torch.cat([input]*(len(timegrid)-1),0)
    tx = torch.cat([t,input],1)
    value_tx = netValue(tx)
    return value_tx




def evaluation_improvement_law(batch_size, base_lr, n_iter, dim, kappa, sigma, init_t, T, timestep):
    batch_size=1000
    base_lr=0.001
    n_iter=500
    dim = 10
    kappa = 4
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    
    # number of players and init_values
    n_players = 30
    if cuda:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim])).cuda()
    else:
        init_values = torch.Tensor(np.random.uniform(low=-1, high=1, size=[n_players, dim]))
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
    netValue = Net_Algorithm2(dim=dim)
    if cuda:
        netValue.cuda()

    timegrid = np.arange(init_t, T+timestep/2, timestep)
    if cuda:
        netValue.cuda()
    if cuda:
        netAlpha.cuda()
    

    it = 0
    laws = [law]
    for it in range(n_iter):
        print('Iteration {}/{}'.format(it, n_iter))
        #######################################
        # Gradient Descent for value function #
        #######################################
        print('gradient descent for value evaluation')
        netValue.train()
        netAlpha.eval()
        evaluation_step(batch_size=batch_size, base_lr=base_lr, 
                        n_iter=50, dim=dim, kappa=kappa, sigma=sigma, 
                        init_t=init_t, T=T, timestep=timestep)
               
        ############################################
        # Gradient descent for policy minimisation #
        ############################################
        print('gradient descent for policy minimisation')
        netValue.eval()
        netAlpha.train()
        policy_improvement_step_new(batch_size=batch_size, base_lr=base_lr, 
                                    n_iter=20, dim=dim, kappa=kappa, 
                                    sigma=sigma, init_t=init_t, T=T, timestep=timestep)
        
        
        ###################
        # law improvement #
        ###################
        

        ########################################
        # now we improve law using Monte Carlo #
        ########################################
        if (it+1)%100==0: 
            law = law_improvement_step(init_values)
            laws.append(law)
    
    
    netAlpha.eval()
    x1 = np.arange(-1,1,0.01)
    xx = []
    policies = []
    for xx1 in x1:
        coord1 = torch.Tensor([xx1]).cuda().view(1,-1)
        x = torch.cat([coord1,torch.zeros(1, dim-1).cuda()],1)
        x = Variable(x)
        xx.append(x)
        pol = torch.zeros([len(timegrid)-1, dim])
        for i in range(len(timegrid)-1):
            t = Variable((torch.ones(1,1)*timegrid[i]).cuda())            
            tx = torch.cat([t,x], dim=1)
            alpha_tx = netAlpha(tx)
            pol[i] = alpha_tx.data.view(-1)

        policies.append(pol)
        
    pol_1st = [p[:,0].contiguous().view(-1,1) for p in policies]  # each row: time. Each column: x
    pol_1st = torch.cat(pol_1st, 1).numpy()  # matrix of shape timegrid x xgrid
    np.savetxt('pol_1st.txt', pol_1st)
    pol_1st = np.loadtxt('pol_1st.txt')
        
    X_grid, Y_grid = np.meshgrid(x1, timegrid[:-1])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot_surface(X_grid, Y_grid, pol_1st, cmap="coolwarm",
                           linewidth=0, antialiased=False)


def algorithm2():
    
    batch_size=500
    base_lr=0.001
    n_iter=200
    dim = 10
    kappa = 4
    sigma = 0.1
    init_t = 0
    T = 1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    eps = 20
    
    # number of players and init_values
    n_players = 20
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
        
    
    law_iterations = 10
    norm_alphas = []
    norm_values = []
    
    netValue = Net_Algorithm2(dim=dim)
    netAlpha = Net_alpha(dim=dim)
    if cuda:
        netAlpha.cuda()
        netValue.cuda()
    
    for law_it in range(law_iterations):
        alpha_converges = False
        norm_alphas = []
            
        # 1. we initialise alpha

        it_law = 0
        while (not alpha_converges) and it_law<5:
            it_law += 1
            # 2. evaluation step: approximation of v to PDE              
            netValue.train()
            netAlpha.eval()   

            evaluation_step(batch_size=500, base_lr=0.01, 
                            n_iter=1000, dim=dim, kappa=kappa, 
                            sigma=sigma, init_t=init_t, T=T, timestep=timestep) 
            a = getNorm_value_from_batch(batch_space_to_measure_alpha)
            norm_values.append(a)
                        
            # 3. Policy optimisation step: we minise alpha
            #netAlpha = Net_alpha(dim=dim)
            if cuda:
                netAlpha.cuda()
            netAlpha.train()
            netValue.eval()
            policy_improvement_step_new(batch_size=4000, base_lr=0.005, n_iter=300, #n_iter = 3000, 
                                    dim=dim, kappa=kappa, sigma=sigma, 
                                    init_t=init_t, T=T, timestep=timestep) 

            netAlpha.eval()
            a = getNorm_alpha_from_batch(batch_space_to_measure_alpha)
            #norm_alphas_iteration.append(a)
            norm_alphas.append(a)
            
            # we check for convergence
            if len(norm_alphas)>1:
                n = torch.norm(norm_alphas[-1]-norm_alphas[-2]).data.cpu().numpy()
                print("norm difference alphas is {:.4f}".format(n[0]))
                if torch.norm(norm_alphas[-1]-norm_alphas[-2])<eps:
                    alpha_converges=True
    
        
        # 4. Law improvement: Monte Carlo to improve the law
        netValue.eval()
        netAlpha.eval()
        law = law_improvement_step(init_values)
        if cuda:
            law = law.cpu().data
            law = Variable(law.cuda())
        else:
            law = law.data
            law = Variable(law)
        laws.append(law)

        
#################### TEST
netValue.eval()
netAlpha.eval()
l = []
t = torch.ones(init_values.size()[0],1)*timegrid[-1]
t = Variable(t.cuda())
tx = torch.cat([t,init_values], 1)
v,_ = netValue(tx)

x = init_values[0].view(1,-1)
t = (torch.ones(1,1)*timegrid[0]).cuda()
t = Variable(t)
tx = torch.cat([t,x], 1)
_,grad = netValue(tx)
alpha = netAlpha(tx)




path = get_path(init_values[0], n_iter=1)
path = torch.cat(path, 0)

def get_path(x, n_iter):
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    netAlpha.eval()
    x = torch.cat([x.view(1,-1)]*n_iter, 0)
    path = [Variable(x.data)]
    
    for i in range(len(timegrid)-1):
        h = timegrid[i+1]-timegrid[i]
        if cuda:
            t = (torch.ones(n_iter,1)*timegrid[i]).cuda()
            xi = Variable(torch.randn(n_iter, dim).cuda())
        else:
            t = torch.ones(n_iter,1)*timegrid[i]
            xi = Variable(torch.randn(n_iter, dim))
            
        t = Variable(t)
        tx = torch.cat([t,x],1)
        alpha_tx = netAlpha(tx)            
        if cuda:
            x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi
        else:
            x = x + (alpha_tx) * h + sigma*math.sqrt(h)*xi # to complete - make it general for any LQR problem
        path.append(Variable(x.data))
    return path # list of length len(timegrid). Each element of the list is a timestep, a matrix of size (n_iter, dim)


