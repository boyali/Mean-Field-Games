import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce
from scipy.integrate import trapz



class Model_PDE_one(nn.Module):
    """
    This only accepts one record at a time
    """
    def __init__(self, b, b_bar, c, q, sigma, b_f, b_f_bar, x_bar, k, x_0, eta, c_f, timegrid):
        super(Model_PDE_one, self).__init__()
        # Layers of the network
        self.i2h1 = nn.Sequential(nn.Linear(2,100,bias=True), 
                                  nn.BatchNorm1d(100), 
                                  nn.Tanh())
        self.i2h2 = nn.Sequential(nn.Linear(2,100, bias=True), nn.BatchNorm1d(100),
                                  nn.Sigmoid())
        #self.h2h = nn.Sequential(nn.Linear(50,50,bias=True), nn.ReLU)
        self.h_o = nn.Linear(100,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.b_bar = b_bar
        self.c = c
        self.q = q
        self.sigma = sigma
        self.b_f = b_f
        self.b_f_bar = b_f_bar
        self.x_bar = x_bar
        self.k = k
        self.x_0 = x_0
        self.eta = eta
        self.c_f = c_f
        self.timegrid = timegrid
        
    def forward(self, x):
        # Design of neural network
        h1 = self.i2h1(x)
        h2 = self.i2h2(x)
        h = h1 * h2
        output = self.h_o(h)
        
        # PDE to calculate the error
        # we get df/dx and df/dt where f(t,x) is the approimation of the
        
        # PDE solution given by our neural network
        l = []
        for row in range(output.size()[0]):
            input_grad = torch.autograd.grad(output[row], x, create_graph=True)[0]
            l.append(input_grad)
        grad_f = reduce(lambda x,y: x+y, l)
        df_dx = grad_f[1]
        df_dt = grad_f[0]
        # we get second derivatives
        grad_df_dx = []    
        for batch in range(df_dx.size()[0]):
            grad_df_dx.append(torch.autograd.grad(df_dx[batch], x, create_graph=True)[0])
        grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
        df_dxdx = grad_df_dx[1]
        df_dxdt = grad_df_dx[0]
        
        #pde = self.b_f*x[1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
        t = np.around(x[0].data.numpy().astype('float64'), decimals=2)
        #print('t={}'.format(float(t)))
        ind = np.where(self.timegrid==float(t))[0]
        eta_t = self.eta[int(ind)]
        x_bar_t = self.x_bar[int(ind)]
        x_0_t = self.x_0[int(ind)]
        #print('ind={}, type ind = {}, len ind = {}'.format(ind, type(ind),len(ind)))
        #alpha_t_x = self.alphas[int(ind)](x[1])
        #pde = self.b_f*x[1]**2 + self.c_f*(-0.1*x[1])**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*(-0.1*x[1]))*df_dx
        #pde = self.b_f*x[1]**2 + self.c_f*(alpha_t_x)**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*alpha_t_x)*df_dx
        alpha = -self.c/(2*self.c_f)*df_dx
        pde = self.c_f*alpha**2 + 0.5*(self.b_f*x[1] - self.b_f_bar*x_bar_t - self.k*self.x_0_t - eta_t)**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1] + self.b_bar*x_bar_t + self.c*alpha + self.q*x_0_t)
        return output, pde, alpha
    
    
    
def sample_timegrid(timegrid, xlim, batch_size):
    xmin, xmax = xlim
    list_points = []
    for t in timegrid[:-1]:
        tt = t*torch.ones([batch_size,1])
        #t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
        x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
        points = torch.cat([tt,x],dim=1)
        list_points.append(points)
    points = torch.cat(list_points, dim=0)
    
    terminal_points_x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    terminal_points_t = torch.ones([batch_size,1])*timegrid[-1]
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points



class Player():
    
    def __init__(self, b, b_bar, c, q, sigma, b_f, b_f_bar, x_bar, k, x_0, eta, c_f, timegrid, xlim, batch_size):
        #Parameters of the player
        self.b = b
        self.b_bar = b_bar
        self.c = c
        self.q = q
        self.sigma = sigma
        self.b_f = b_f
        self.b_f_bar = b_f_bar
        self.x_bar = x_bar
        self.k = k
        self.x_0 = x_0
        self.eta = eta
        self.c_f = c_f
        self.timegrid = timegrid
        self.xlim = xlim
        self.batch_size = batch_size
        self.law = []
    
    
    def _init_data_batch(self):
        data_batch, terminal_points = sample_timegrid(self.timegrid, self.xlim, self.batch_size)
        self.data_batch = Variable(data_batch, requires_grad=True)
        self.terminal_points = Variable(terminal_points, requires_grad=True)
        
        # the target for the loss function is a vector of zeros, and terminal values        
        self.target = Variable(torch.zeros(self.data_batch.size()[0]))
        self.target_terminal = Variable(self.gamma*self.terminal_points.data[:,1]**2)
    
    
    
    def _init_value_function(self):
        self.value_function = Model_PDE_one(b=self.b, b_bar=self.b_bar, c=self.c, 
                                       q=self.q, sigma=self.sigma, b_f=self.b_f, 
                                       b_f_bar=self.b_f_bar, x_bar=self.x_bar, k=self.k, 
                                       x_0=self.x_0, eta=self.eta, c_f=self.c_f, timegrid=self.timegrid)
    
    def solve_HJB(self):
        self._init_value_function()
        criterion = nn.MSELoss()
        base_lr = 0.9
        n_iter = 4
        optimizer = torch.optim.LBFGS(self.value_function.parameters(),lr=base_lr, max_iter=10)    
        
        for it in range(n_iter):
            
            def closure():
                optimizer.zero_grad()
                # the output of the model is solution of pde, and the pde itself for the loss function
                PDE = []
                for row in range(self.data_batch.size()[0]):
                    data_batch_record = self.data_batch[row]
                    _, PDE_record = self.value_function(data_batch_record)
                    PDE.append(PDE_record)
                PDE = torch.cat(PDE)
                output_terminal = []
                for row in range(self.terminal_points.size()[0]):
                    terminal_batch_record = self.terminal_points[row]
                    output_terminal_record, _ = self.value_function(terminal_batch_record)
                    output_terminal.append(output_terminal_record)
                output_terminal = torch.cat(output_terminal)            
                loss = criterion(PDE, self.target) + criterion(output_terminal, self.target_terminal)    
                loss.backward()
                print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
                return loss
            lr = base_lr * (0.8 ** (it // 2))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
            optimizer.step(closure)
    
    
    def generate_path(self, y0):
        
    
    
    def update_law(self, pol):
        # do several simulations. We cannot solve the ode here because alpha is not linear on x anymore
        
        p1_grid = pol.p1_grid[-1]
        p2_grid = pol.p2_grid[-1]
        p3_grid = np.array([self.p3(t) for t in self.time])
        p2_grid = p2_grid - p3_grid * self.law[-1]
        
        a0 = self.c*p2_grid 
        a1 = self.b + self.c*p1_grid + self.c*p3_grid + self.m
        
        y_0 = self.law_0
        #current_law = self._solve_ode_euler(a0,a1,y_0)
        
        current_law = self._solve_ode_explicit(a0,a1,y_0)
        self.law.append(np.array(current_law))
    
    
    
    
    def _solve_ode_explicit(self, a0, a1, y_0):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = gamma
        withi initial condition y(0)
        
        Input:
            - a0: function
            - a1: function
            - y_0: value of y(0)
        
        Output:
            - solution of ODE at time grid points. 
            
        Note: we use Simpson's rule to solve the integrals
        Note: This ffunction still doesn't work. Need to be corrected
        """
        a0 = np.array(a0)
        a1 = np.array(a1)
        val_integral1 = np.zeros_like(self.time)
        val_integral2 = np.zeros_like(self.time)
        val_integral3 = np.zeros_like(self.time)
        
        # we fill val_integral1
        for i in range(1, len(self.time)):
            integrand1 = -np.copy(a1[i-1:i+1])
            val_integral1[i] = trapz(integrand1, self.time[i-1:i+1])
        val_integral1 = np.cumsum(val_integral1)
        e_integral1 = np.exp(-val_integral1)
        
        # we fill val_integral3
        for i in range(1, len(self.time)):
            val_integral3[i] = trapz(-a1[i-1:i+1], self.time[i-1:i+1])
        val_integral3 = np.cumsum(val_integral3)
        
        # we get integrand2
        integrand2 = a0 * np.exp(val_integral3)
        for i in range(1, len(self.time)):
            val_integral2[i] = trapz(integrand2[i-1:i+1], self.time[i-1:i+1])
        val_integral2 = np.cumsum(val_integral2) 
        
        solution_ode = e_integral1*(y_0 + val_integral2)
        return solution_ode
        

        
        

        

