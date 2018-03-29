"""
This script solves the MFG linear quadratic game with 
one major player, and several minor players.

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
from numpy.linalg import norm



class Net_stacked_major(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the major player are given in the report. 
    """
    
    def __init__(self, dim, b, b_bar, x_bar, c, sigma, b_f, b_f_bar, eta, c_f, timegrid):
        super(Net_stacked_major, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.b = b
        self.b_bar = b_bar
        self.x_bar = x_bar  # x_bar is the law. It should be an array of values. 
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.b_f_bar = b_f_bar
        self.c_f = c_f
        self.eta = eta # eta should be a lambda function
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
        path = [x]
        dW = []
        grad_path = []
        for i in range(0,len(self.timegrid)-1):
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = torch.sqrt(h)*Variable(torch.randn(x.data.size()))
            #print('i={}\t x.size={}\t xi.size={}'.format(i,x.data.size(),xi.data.size()))

            dW.append(xi)
            
            # we update value function
            
            if i == 0:
                grad_path.append(self.grad_v0)
                alpha = -self.c/self.c_f * self.grad_v0
                f = 0.5*(self.b_f*x - self.b_f_bar*self.x_bar[i] - self.eta[i])**2 + 0.5*self.c_f*alpha**2
                v = self.v0 - f*h + self.grad_v0*self.sigma*xi
            else:
                h1 = self.i_h1[i-1](x)
                h2 = self.h1_h2[i-1](h1)
                grad = self.h2_o[i-1](h2)
                grad_path.append(grad)
                alpha = -self.c/self.c_f * grad
                f = 0.5*(self.b_f*x - self.b_f_bar*self.x_bar[i] - self.eta[i])**2 + 0.5*self.c_f*alpha**2
                v = v - f*h + grad*self.sigma*xi
            
            # we update x
            #x = x + (self.b*x + self.c*alpha) * h + self.sigma*xi
            x = x + (self.b*x + self.b_bar*self.x_bar[i-1] + self.c*alpha) * h + self.sigma*xi
            path.append(x)
        
        return v, x, path, grad_path, dW
    
    
class Net_stacked_minor(nn.Module):
    """
    We create a network that approximates the solution of
    v(0,xi) of the HJB PDE equation with terminal condition
    Reference paper: https://arxiv.org/pdf/1706.04702.pdf
    
    The dynamics of the minor players are given in the report. 
    """
    
    def __init__(self, dim, b_minor, b_bar_minor, x_bar, c_minor, q_minor, 
                 b_major, b_bar_major, c_major,
                 sigma, b_f, k, b_f_bar, eta, c_f, timegrid,
                 net_major):
        super(Net_stacked_minor, self).__init__()
        self.dim = dim
        self.timegrid = Variable(torch.Tensor(timegrid))
        self.b_minor = b_minor
        self.k = k
        self.b_bar_minor = b_bar_minor
        self.x_bar = x_bar  # x_bar is the law. It should be an array of values. 
        self.c_minor = c_minor
        self.q_minor = q_minor
        self.sigma = sigma
        self.b_major = b_major
        self.b_bar_major = b_bar_major
        self.c_major = c_major
        self.b_f = b_f
        self.b_f_bar = b_f_bar
        self.c_f = c_f
        self.net_major = net_major
        self.eta = eta # eta should be a lambda function
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
        
        # we get a major path
        self.net_major.eval()
        input_major = Variable(torch.ones([1, 1])*0)
        _, _, path_major, grad_major, dW_major = self.net_major(input_major)   
        
        # we do this to avoid complications in the optimisation step, so that we don't optimise parameters of the major player
        path_major = [Variable(xx.data, requires_grad = False) for xx in path_major]
        grad_major = [Variable(xx.data, requires_grad = False) for xx in grad_major]
        path = [x]
        
        for i in range(0,len(self.timegrid)-1):
            
            h = self.timegrid[i+1]-self.timegrid[i]
            xi = torch.sqrt(h)*Variable(torch.randn(self.dim))
            #print('i={}\t x.size={}\t xi.size={}'.format(i,x.data.size(),xi.data.size()))
            
            
            # we update value function
            if i == 0:
                alpha = -self.c_minor/self.c_f * self.grad_v0
                f = 0.5*(self.b_f*x-self.b_bar_minor*self.x_bar[i]-self.k*path_major[i]-self.eta[i])**2 + 0.5*self.c_f*alpha**2
                v = self.v0 - f*h + self.grad_v0*self.sigma*xi + grad_major[i]*self.sigma*dW_major[i]
            else:
                h1 = self.i_h1[i-1](x)
                h2 = self.h1_h2[i-1](h1)
                grad = self.h2_o[i-1](h2)
                alpha = -self.c_minor/self.c_f * grad
                f = 0.5*(self.b_f*x-self.b_bar_minor*self.x_bar[i]-self.k*path_major[i]-self.eta[i])**2 + 0.5*self.c_f*alpha**2
                v = v - f*h + grad*self.sigma*xi + grad_major[i]*self.sigma*dW_major[i]
            
            # we update x
            x = (self.b_minor*x + self.q_minor*path_major[i] + self.b_bar_minor*self.x_bar[i] + self.c_minor*alpha)*h + self.sigma*xi
            path.append(x)
            
        return v, x, path
    
    
class major_minor_LQR_MFG():
    """
    This class solves a MFG:
    Algorithm:
        - We fix the law and find the major player strategies, using the DL approximation of the value function
        - We fix the major player strategy, and find the minor players strategies, using the DL approximation of the value function
        - We update the law.    
    """
    
    def __init__(self,b_major, b_bar_major, c_major, sigma, b_f_major, b_f_bar_major, eta_major, c_f_major, gamma_major, gamma_bar_major,
                 b_minor, k, b_bar_minor, c_minor, q_minor, b_f_minor, b_f_bar_minor, eta_minor, c_f_minor, gamma_minor, gamma_bar_minor,
                 init_t, T, timestep):
        self.b_major = b_major
        self.b_bar_major = b_bar_major
        self.c_major = c_major
        self.sigma = sigma
        self.b_f_major = b_f_major
        self.b_f_bar_major = b_f_bar_major
        self.eta_major = eta_major
        self.c_f_major = c_f_major
        self.gamma_major = gamma_major
        self.gamma_bar_major = gamma_bar_major
        self.b_minor = b_minor
        self.k = k
        self.b_bar_minor = b_bar_minor
        self.c_minor = c_minor
        self.q_minor = q_minor
        self.b_f_minor = b_f_minor
        self.b_f_bar_minor = b_f_bar_minor
        self.eta_minor = eta_minor
        self.c_f_minor = c_f_minor
        self.gamma_minor = gamma_minor
        self.gamma_bar_minor = gamma_bar_minor
        self.init_t = init_t
        self.T= T
        self.timestep = timestep
        self.timegrid = np.arange(init_t, T+timestep/2, timestep)
        self._init_x_bar()
        self.major_player_value = []
        self.minor_player_value = []
        
    def _init_x_bar(self):
        self.x_bar = np.zeros_like(self.timegrid)
        
    def major_player(self, batch_size=200, n_iter=100):
        model = Net_stacked_major(dim=1, b=self.b_major, b_bar=self.b_bar_major, 
                                  x_bar=self.x_bar, c=self.c_major, sigma=self.sigma, 
                                  b_f=self.b_f_major, b_f_bar=self.b_f_bar_major, eta=self.eta_major, 
                                  c_f=self.c_f_major, timegrid=self.timegrid)
        model.train()
        #batch_size = 120
        base_lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        criterion = torch.nn.MSELoss()
        
        #n_iter = 100
        #v0 = []
        
        for it in range(n_iter):
            lr = base_lr * (0.5 ** (it // 25))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
            optimizer.zero_grad()
            x0 = 0
            input = torch.ones([batch_size, 1])*x0
            input = Variable(input)
            output, x_T, _, _, _ = model(input)
            target = 0.5*(self.gamma_major*x_T-self.gamma_bar_major*self.x_bar[-1])**2
            target = Variable(target.data, requires_grad=False)  # we don't want to create a loop, as alpha also depends on the parameters, and target depends on alpha
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            #v0.append(copy.deepcopy(model.state_dict()['v0'].numpy())[0])
        
        self.major_player_value.append(model)
        return model
    
    def minor_player(self, major_player_model, batch_size=200, n_iter=100):
        model = Net_stacked_minor(dim=1, b_minor=self.b_minor, b_bar_minor=self.b_bar_minor, 
                                  x_bar=self.x_bar, c_minor=self.c_minor, q_minor=self.q_minor,
                                  b_major=self.b_major, b_bar_major=self.b_bar_major, c_major=self.c_major,
                                  sigma=self.sigma, b_f=self.b_f_minor, k=self.k, b_f_bar=self.b_f_bar_minor, 
                                  eta=self.eta_minor, c_f=self.c_f_minor, timegrid=self.timegrid,
                                  net_major = major_player_model)
        model.train()
        #batch_size = 120
        base_lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        criterion = torch.nn.MSELoss()
        
        #n_iter = 100
        #v0 = []
        
        for it in range(n_iter):
            lr = base_lr * (0.5 ** (it // 25))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
            optimizer.zero_grad()
            x0 = 0
            input = torch.ones([batch_size, 1])*x0
            input = Variable(input)
            output, x_T, _ = model(input)
            target = 0.5*(self.gamma_minor*x_T-self.gamma_bar_minor*self.x_bar[-1])**2
            target = Variable(target.data, requires_grad=False) # we don't want to create a loop, as alpha also depends on the parameters, and target depends on alpha
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print("Iteration=[{it}/{n_iter}]\t loss={loss:.3f}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
        
        self.minor_player_value.append(model)
        return model
    
    def update_law(self, minor_player_model, n_simulations = 100):
        minor_player_model.eval()
        sims = []
        
        for simul in range(n_simulations):
            if simul%50 == 0:
                print('updating law... Simulation {}/{}'.format(simul, n_simulations))
            x0 = 0
            input = torch.ones([1, 1])*x0
            input = Variable(input)
            _, _, x_minor = minor_player_model(input)
            x_minor = np.concatenate([x.data[0].numpy() for x in x_minor])
            sims.append(x_minor)
        sims = np.array(sims)
        sims = np.apply_along_axis(np.mean, axis=0, arr=sims)
        self.x_bar = sims


def main():
    """
    MFG LQR using Deep Learning (OH GOD)
    """
    b_major = 0
    b_bar_major=0 
    c_major = 1
    sigma=0.05
    b_f_major = 1
    b_f_bar_major = 0.5
    init_t = 0
    T = 5
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    eta_major = (1/(1+np.exp(-timegrid))-0.5)*6 #np.sin(timegrid*np.pi)
    c_f_major=0.5
    gamma_major = 0#1
    gamma_bar_major = 0#0.1
    b_minor = 0
    k = 0.8
    b_bar_minor = 0.5
    c_minor = 1
    q_minor = 1.5
    b_f_minor = 1
    b_f_bar_minor = 0.8
    eta_minor = (1/(1+np.exp(-timegrid))-0.5)*0.5
    c_f_minor = 1
    gamma_minor = 0#1
    gamma_bar_minor = 0#0.5
    law = []
    game = major_minor_LQR_MFG(b_major=b_major, b_bar_major=b_bar_major, c_major=c_major, sigma=sigma, 
                               b_f_major=b_f_major, b_f_bar_major=b_f_bar_major, eta_major=eta_major, 
                               c_f_major=c_f_major, gamma_major=gamma_major, gamma_bar_major=gamma_bar_major, 
                               b_minor=b_minor, k=k, b_bar_minor=b_bar_minor, c_minor=c_minor, q_minor=q_minor, 
                               b_f_minor=b_f_minor, b_f_bar_minor=b_f_bar_minor, eta_minor=eta_minor, 
                               c_f_minor=c_f_minor, gamma_minor=gamma_minor, gamma_bar_minor=gamma_bar_minor, 
                               init_t=init_t, T=T, timestep=timestep)
    
    law.append(game.x_bar)
    for i in range(10):
    
        major_player_model = game.major_player(batch_size=900,n_iter=450)
        minor_player_model = game.minor_player(major_player_model, batch_size=900,n_iter=100)
        game.update_law(minor_player_model, n_simulations=500)
        law.append(game.x_bar)
    
    plt.plot(game.x_bar)
    plt.plot(game.eta_major)
    plt.plot(game.eta_minor)
    
    # norm differences of the law
    [norm(law[i]-law[i-1]) for i in range(1,len(law))]

#    
#    x0 = 0
#    input = torch.ones([1, 1])*x0
#    input = Variable(input)
#    v, x, path = minor_player_model(input)
#    x_minor = np.concatenate([x.data[0].numpy() for x in path])
#    plt.plot(x_minor)
#    
#    v, x, path, grad_path, dW = major_player_model(input)
#    x_major = np.concatenate([x.data[0].numpy() for x in path])
#    plt.plot(x_major)
#    plt.plot(game.eta_major)
    
    
    # plots results
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(timegrid, x_major, '-', label='major player state')
    ax1.plot(timegrid, game.eta_major, label='major player goal')
    #ax1.plot(timegrid, x_minor, label='minor player state')
    ax1.legend()
    ax1.set_xlabel('time')
    fig.savefig('major_player_state_MFG_LQR.png')
    
    
    x_minor = []
    for i in range(5):
        v, x, path = minor_player_model(input)
        x_minor.append(np.concatenate([x.data[0].numpy() for x in path]))
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i in range(3):
        ax2.plot(timegrid, x_minor[i], label='player '+str(i+1))
    ax2.legend()
    ax2.set_xlabel('time')
    ax2.set_ylabel('minor players state')
    fig2.savefig('minor_player_state_MFG_LQR.png')
    
    
    
if __name__=='__main__':
    main()
    
    

    
        
            
            
        
        
        
        
        

        
        
    
        
        
        
        
        
        