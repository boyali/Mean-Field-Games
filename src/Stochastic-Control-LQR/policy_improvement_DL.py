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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, it, is_best):
    filename = 'checkpoint_w_it' + str(it) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_long.pth.tar')

class Model_PDE_one(nn.Module):
    """
    This only accepts one record at a time
    """
    def __init__(self, b, c, sigma, b_f, c_f, alphas, timegrid):
        super(Model_PDE_one, self).__init__()
        # Layers of the network
        self.i2h1 = nn.Sequential(nn.Linear(2,100,bias=True), nn.Tanh())
        self.i2h2 = nn.Sequential(nn.Linear(2,100, bias=True), nn.Sigmoid())
        #self.h2h = nn.Sequential(nn.Linear(50,50,bias=True), nn.ReLU)
        self.h_o = nn.Linear(100,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.alphas = alphas
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
        #print('ind={}, type ind = {}, len ind = {}'.format(ind, type(ind),len(ind)))
        alpha_t_x = self.alphas[int(ind)](x[1])
        #pde = self.b_f*x[1]**2 + self.c_f*(-0.1*x[1])**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*(-0.1*x[1]))*df_dx
        pde = self.b_f*x[1]**2 + self.c_f*(alpha_t_x)**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*alpha_t_x)*df_dx
        return output, pde


class Model_alpha(nn.Module):
    
    def __init__(self):
        super(Model_alpha, self).__init__()
        self.i_h1 = nn.Sequential(nn.Linear(1,10), nn.Tanh())
        self.h1_o = nn.Linear(10,1, bias=True)
        
    def forward(self,x):
        h1 = self.i_h1(x)
        output = self.h1_o(h1)
        return output


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




class PolicyIteration_DL():
    
    def __init__(self, init_t, T, timestep, xlim, batch_size, b, c, sigma, b_f, c_f, gamma):
        self.init_t = init_t
        self.T = T
        self.timestep = timestep
        self.timegrid = np.around(np.arange(init_t, T+timestep/2, timestep), decimals=2)
        self.xlim = xlim
        self.batch_size = batch_size
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        self._init_alphas()
        self._init_data_batch()
        
    
    def _init_data_batch(self):
        data_batch, terminal_points = sample_timegrid(self.timegrid, self.xlim, self.batch_size)
        self.data_batch = Variable(data_batch, requires_grad=True)
        self.terminal_points = Variable(terminal_points, requires_grad=True)
        
        # the target for the loss function is a vector of zeros, and terminal values        
        self.target = Variable(torch.zeros(self.data_batch.size()[0]))
        self.target_terminal = Variable(self.gamma*self.terminal_points.data[:,1]**2)

    
    def _init_alphas(self):
        self.alphas = [Model_alpha() for t in self.timegrid]
        
    def _init_value_function(self):
        self.value_function = Model_PDE_one(b=self.b, c=self.c, sigma=self.sigma, b_f=self.b_f, c_f=self.c_f, alphas=self.alphas, timegrid=self.timegrid)
    
    def get_value_function(self):
        output = []
        for row in range(self.data_batch.size()[0]):
            data_batch_record = self.data_batch[row]
            val, _ = self.value_function(data_batch_record)
            output.append(val)
        for row in range(self.terminal_points.size()[0]):
            terminal_batch_record = self.terminal_points[row]
            output_terminal_record, _ = self.value_function(terminal_batch_record)
            output.append(output_terminal_record)
        output = torch.cat(output)
        output = output.data.numpy()
        return output
    
    def get_alpha(self):
        output = []
        for row in range(self.data_batch.size()[0]):
            x = self.data_batch[row]
            t = np.around(x[0].data.numpy().astype('float64'), decimals=2)
            #print('t={}'.format(float(t)))
            ind = np.where(self.timegrid==float(t))[0]
            #print('ind={}, type ind = {}, len ind = {}'.format(ind, type(ind),len(ind)))
            alpha_t_x = self.alphas[int(ind)](x[1])
            output.append(alpha_t_x)
        for row in range(self.terminal_points.size()[0]):
            x = self.data_batch[row]
            alpha_t_x = self.alphas[-1](x[1])
            output.append(alpha_t_x)
        output = torch.cat(output)
        output = output.data.numpy()
        return output
            
    def evaluation_step(self):
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
    
    def improvement_step(self, epochs = 20):
        self._init_alphas()
        for ind in range(len(self.timegrid)):
            # we train alpha_t
            base_lr = 0.001
            optimizer = torch.optim.SGD(self.alphas[ind].parameters(),lr=base_lr, momentum=0.9)
            ind_t = self.data_batch.data[:,0]==self.timegrid[ind]
            ind_t = np.where(ind_t.numpy()==1)[0]
            if ind == len(self.timegrid)-1:
                data_batch_t = self.terminal_points
            else:
                data_batch_t = self.data_batch[ind_t,:]
            for epoch in range(epochs):
                lr = base_lr * (0.5 ** (epoch // 5))
                for param_group in optimizer.state_dict()['param_groups']:
                    param_group['lr'] = lr
                for row in range(data_batch_t.size()[0]):
                    optimizer.zero_grad()
                    x = data_batch_t[row]
                    xx = data_batch_t[row][1]
                    value, _ = self.value_function(x)
                    df_dx = torch.autograd.grad(value, x, create_graph=True)[0]
                    df_dx = df_dx[1]
                    alpha_x = self.alphas[ind](xx) 
                    loss = self.c_f*alpha_x**2 + self.c*alpha_x*df_dx
                    print('Time: {:.3f}\t epoch: {}\t row: {}\t loss: {:.3f}'.format(self.timegrid[ind],epoch,row,loss.data[0]))
                    loss.backward()
                    optimizer.step()
        return 1
                    
                    
                    
# TEST
init_t, T = 0,1
timestep = 0.05
xlim = (0,5)
b=0.5 
c=0.5 
sigma=1 
b_f=0.5 
c_f=0.9
gamma = 1
n_iter = 10

pol_DL = PolicyIteration_DL(init_t = init_t, T=T, timestep=timestep, xlim=xlim, batch_size = 20, b=b, c=c, sigma=sigma, 
                         b_f=b_f, c_f=c_f, gamma=gamma)
alphas_DL = []
alphas_DL.append(pol_DL.get_alpha())
for it in range(n_iter):
    pol_DL.evaluation_step()
    pol_DL.improvement_step()
    alphas_DL.append(pol_DL.get_alpha())

   
diff_alphas = [norm(alphas_DL[i+1]-alphas_DL[i]) for i in range(len(alphas_DL)-1)]        

# we want to compare this solution with the explicit solution 
# we will need to make some changes in how we compute the 
# TODO: add original policy iteration here

data = pol_DL.data_batch.data.numpy().astype('float64')
terminal_data = pol_DL.terminal_points.data.numpy().astype('float64')
data = np.concatenate([data, terminal_data], axis=0)
data[:,0] = np.around(data[:,0], decimals=2)

hjb = HJB_LQR(b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, gamma=gamma, T=T, init_t = init_t, solve_ode=True, timestep=0.00001)
hjb.time = np.around(hjb.time, decimals=5)
alpha_LQR_sol = []
for row in data[:]:
    t,x = row[0], row[1]
    ind_t = np.where(hjb.time==t)[0]
    alpha_LQR = hjb.get_alpha(x)[ind_t]
    alpha_LQR_sol.append(alpha_LQR[0])

alpha_LQR_sol = np.array(alpha_LQR_sol)
alpha_LQR_sol[-10:]
alphas_DL[0][-10:]


norms = np.array([1/alpha_DL.shape[0]*norm(alpha_LQR_sol-alpha_DL)**2 for alpha_DL in alphas_DL])
iteration = np.arange(len(norms))
fig =plt.figure()
ax = fig.add_subplot(111)
ax.plot(iteration, norms, 'b-o')
ax.set_xlabel('iteration')
ax.set_ylabel('Mean Squared Error')
plt.show()
fig.savefig('DL_MSE.png')    

