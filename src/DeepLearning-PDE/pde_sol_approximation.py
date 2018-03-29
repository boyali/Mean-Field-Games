import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from functools import reduce
import shutil
import time

def test_hessian():
    """
    Very simple example on how to obtain second derivatives of f(x,t,theta)
        - df/dt
        - d^2f/dxdx
    We will need this to build the cost function for the model
    that approximates the solution of a parabolic PDE
    
    """
    x = Variable(torch.randn(2), requires_grad=True)
    x = Variable(torch.Tensor([1,2]), requires_grad=True)
    theta = Variable(torch.Tensor([1]), requires_grad=True)
    y = x**2
    z = y.sum() + theta*x[0]*x[1]
    
    input_grad = torch.autograd.grad(z, x, create_graph=True)[0]
    hessian = [torch.autograd.grad(input_grad[i], x, create_graph=True)[0] for i in range(x.size()[0])]
    
    J = input_grad[0] - hessian[1][1]
    
    torch.autograd.grad(J, theta, create_graph=True)[0]
    
    return 1


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


class Model_PDE(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=0.1):
        super(Model_PDE, self).__init__()
        # Layers of the network
        self.i2h1 = nn.Sequential(nn.Linear(2,100,bias=True), nn.Sigmoid())
        self.i2h2 = nn.Sequential(nn.Linear(2,100, bias=True), nn.Sigmoid())
        #self.h2h = nn.Sequential(nn.Linear(50,50,bias=True), nn.ReLU)
        self.h2o = nn.Linear(100,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        
    def forward(self, x):
        # Design of neural network
        h1 = self.i2h1(x)
        h2 = self.i2h2(x)
        h = h1 * h2
        output = self.h2o(h)
        
        # PDE to calculate the error
        # we get df/dx and df/dt where f(t,x) is the approimation of the
        
        # PDE solution given by our neural network
        l = []
        for row in range(output.size()[0]):
            input_grad = torch.autograd.grad(output[row], x, create_graph=True)[0]
            l.append(input_grad)
        grad_f = reduce(lambda x,y: x+y, l)
        df_dx = grad_f[:,1]
        df_dt = grad_f[:,0]
        # we get second derivatives
        grad_df_dx = []    
        for batch in range(df_dx.size()[0]):
            grad_df_dx.append(torch.autograd.grad(df_dx[batch], x, create_graph=True)[0])
        grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
        df_dxdx = grad_df_dx[:,1]
        df_dxdt = grad_df_dx[:,0]
        
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
        return output, pde
    
class Model_PDE_one(nn.Module):
    """
    This only accepts one record at a time
    """
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, alphas, timegrid):
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
        t = x[0].data.numpy()
        ind = np.where(self.timegrid==t)[0]
        alpha_t_x = self.alphas[t](x[1])
        #pde = self.b_f*x[1]**2 + self.c_f*(-0.1*x[1])**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*(-0.1*x[1]))*df_dx
        pde = self.b_f*x[1]**2 + self.c_f*(alpha_t_x)**2 + df_dt + 0.5*self.sigma**2*df_dxdx + (self.b*x[1]+self.c*alpha_t_x)*df_dx
        return output, pde
    
    
class Model_PDE_2(nn.Module):
    
    def __init__(self, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=0.1):
        super(Model_PDE_2, self).__init__()
        # Layers of the network
        self.i2h = nn.Sequential(nn.Linear(2,100,bias=True), nn.Tanh())
        self.h2o = nn.Linear(100,1,bias=True)
        
        
        #Parameters of the PDE
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        
    def forward(self, x):
        # Design of neural network
        h = self.i2h(x)
        output = self.h2o(h)
        
        # PDE to calculate the error
        # we get df/dx and df/dt where f(t,x) is the approimation of the
        
        # PDE solution given by our neural network
        l = []
        for row in range(output.size()[0]):
            input_grad = torch.autograd.grad(output[row], x, create_graph=True)[0]
            l.append(input_grad)
        grad_f = reduce(lambda x,y: x+y, l)
        df_dx = grad_f[:,1]
        df_dt = grad_f[:,0]
        # we get second derivatives
        grad_df_dx = []    
        for batch in range(df_dx.size()[0]):
            grad_df_dx.append(torch.autograd.grad(df_dx[batch], x, create_graph=True)[0])
        grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
        df_dxdx = grad_df_dx[:,1]
        df_dxdt = grad_df_dx[:,0]
        
        pde = self.b_f*x[:,1]**2 + df_dt + 0.5*self.sigma**2*df_dxdx + self.b*x[:,1]*df_dx - self.c**2/(4*self.c_f)*(df_dx)**2
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




def sample(time_interval, xlim, batch_size, terminal_size):
    """
    This function samples batch_size points and terminal_size terminal points 
    """
    xmin, xmax = xlim
    start_time,end_time = time_interval
    
    t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
    x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    points = torch.cat([t,x],dim=1)
    
    terminal_points_x = xmin + torch.rand([terminal_size, 1])*(xmax-xmin)
    terminal_points_t = torch.ones_like(terminal_points_x)*end_time
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points


def sample_timegrid(time_grid, xlim, batch_size):
    """
    This function samples points (t,x) where t\in timegrid and x\in xlim
    
    """
    xmin, xmax = xlim
    start_time,end_time = time_interval
    list_points = []
    for t in time_grid[:-1]:
        tt = t*torch.ones([batch_size,1])
        #t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
        x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
        points = torch.cat([tt,x],dim=1)
        list_points.append(points)
    points = torch.cat(list_points, dim=0)
    
    terminal_points_x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    terminal_points_t = torch.ones([batch_size,1])*time_grid[-1]
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points



def make_dataset(time_interval, xlim, batch_size, fraction_terminal):
    assert(batch_size % fraction_terminal == 0)
    terminal_size = batch_size // fraction_terminal
    xmin, xmax = xlim
    start_time,end_time = time_interval
    
    t = start_time + torch.rand([batch_size, 1])*(end_time-start_time)
    x = xmin + torch.rand([batch_size, 1])*(xmax-xmin)
    points = torch.cat([t,x],dim=1)
    
    terminal_points_x = xmin + torch.rand([terminal_size, 1])*(xmax-xmin)
    terminal_points_t = torch.ones_like(terminal_points_x)*end_time
    terminal_points = torch.cat([terminal_points_t,terminal_points_x], dim=1)
    
    return points, terminal_points
         
        
def train_SGD():
    """
    Training of the network using SGD
    """
    total_size = 5000
    fraction_terminal = 10
    time_interval = (0,1)
    xlim = (8,10)
    gamma = 0.5
    model = Model_PDE_5()
    criterion = nn.MSELoss()
    base_lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum = 0.9)
    
    losses = AverageMeter()
    
    n_epochs = 10
    batch_size = 100
    assert(total_size % batch_size == 0)
    assert(batch_size % fraction_terminal == 0)
    batch_terminal_size = batch_size // fraction_terminal
    data, terminal = make_dataset(time_interval, xlim, total_size, fraction_terminal)
    
    
    for epoch in range(n_epochs):
        data_epoch, terminal_epoch = data[torch.randperm(total_size),:], terminal[torch.randperm(total_size//fraction_terminal),:]
        
        for i in range(total_size//batch_size):
            lr = base_lr * (0.8 ** ((total_size//batch_size * epoch + i) // 5))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
        
            optimizer.zero_grad()
            data_batch = Variable(data_epoch[(i*batch_size):(i*batch_size+batch_size),:], requires_grad=True)    
            terminal_batch = Variable(terminal_epoch[(i*batch_terminal_size):((i+1)*batch_terminal_size),:], requires_grad=True)
            target = Variable(torch.zeros(batch_size))
            target_terminal = Variable(gamma*terminal_batch.data[:,1]**2)
            PDE = []
            for row in range(data_batch.size()[0]):
                data_batch_record = data_batch[row]
                _, PDE_record = model(data_batch_record)
                PDE.append(PDE_record)
            PDE = torch.cat(PDE)
            output_terminal = []
            for row in range(terminal_batch.size()[0]):
                terminal_batch_record = terminal_batch[row]
                output_terminal_record, _ = model(terminal_batch_record)
                output_terminal.append(output_terminal_record)
            output_terminal = torch.cat(output_terminal)
            
            loss = criterion(PDE, target) #+ criterion(output_terminal, target_terminal)
            losses.update(loss.data[0])
            
            # we get the gradients of the loss function by backpropagation
            loss.backward()
            # optimization step
            optimizer.step()
            
            print("iteration: [{it}]\t epoch: {ep}\t loss: {loss:.3f}\t avg_loss: {avg_loss:.3f}".format(it=i, ep=epoch, loss=losses.val, avg_loss=losses.avg))  


def train_LBFGS():
    """
    Training of the network using L-BFGS
    """
    total_size = 20
    fraction_terminal = 10
    time_interval = (0,1)
    xlim = (0,2)
    gamma = 0.5
    model = Model_PDE_one()
    criterion = nn.MSELoss()
    base_lr = 0.001
    n_iter = 10
    optimizer = torch.optim.LBFGS(model.parameters(),lr=0.9, max_iter=30)  
    
    losses = AverageMeter()
    
    data_batch, terminal_batch = make_dataset(time_interval, xlim, total_size, fraction_terminal)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_batch = Variable(terminal_batch, requires_grad=True)

    target = Variable(torch.zeros(total_size))
    target_terminal = Variable(gamma*terminal_batch.data[:,1]**2)
    
    for it in range(n_iter):
        
        def closure():
            optimizer.zero_grad()
            # the output of the model is solution of pde, and the pde itself for the loss function
            PDE = []
            for row in range(data_batch.size()[0]):
                data_batch_record = data_batch[row]
                _, PDE_record = model(data_batch_record)
                PDE.append(PDE_record)
            PDE = torch.cat(PDE)
            output_terminal = []
            for row in range(terminal_batch.size()[0]):
                terminal_batch_record = terminal_batch[row]
                output_terminal_record, _ = model(terminal_batch_record)
                output_terminal.append(output_terminal_record)
            output_terminal = torch.cat(output_terminal)
#            _, PDE = model(data_batch)
#            output_terminal, _ = model(terminal_batch)
            loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    
            loss.backward()
            print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            return loss
        optimizer.step(closure)
    

def train_LBFGS_one():
    """
    training of the network using L-BFGS
    """
    batch_size = 20
    init_t, T = 0,1
    timestep = 0.05
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #time_interval = (0,1)
    xlim = (0,5)
    gamma = 2
    model = Model_PDE_one()
    criterion = nn.MSELoss()
    base_lr = 0.9
    n_iter = 15
    optimizer = torch.optim.LBFGS(model.parameters(),lr=base_lr, max_iter=10)    
    
    # we load all the data
    #data_batch, terminal_points = sample(time_interval, xlim, batch_size, 20)
    data_batch, terminal_points = sample_timegrid(timegrid, xlim, batch_size)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    
    # the target for the loss function is a vector of zeros, and terminal values        
    target = Variable(torch.zeros(data_batch.size()[0]))
    target_terminal = Variable(gamma*terminal_points.data[:,1]**2)
    
    for it in range(n_iter):
        
        def closure():
            optimizer.zero_grad()
            # the output of the model is solution of pde, and the pde itself for the loss function
            PDE = []
            for row in range(data_batch.size()[0]):
                data_batch_record = data_batch[row]
                _, PDE_record = model(data_batch_record)
                PDE.append(PDE_record)
            PDE = torch.cat(PDE)
            output_terminal = []
            for row in range(terminal_points.size()[0]):
                terminal_batch_record = terminal_points[row]
                output_terminal_record, _ = model(terminal_batch_record)
                output_terminal.append(output_terminal_record)
            output_terminal = torch.cat(output_terminal)            
            loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    
            loss.backward()
            print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            return loss
        lr = base_lr * (0.8 ** (it // 2))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
            
        optimizer.step(closure)
    return model


def train_alpha(value_function, timegrid, data_batch, terminal_points, c, c_f):
    epochs = 20
    alphas = [Model_alpha() for t in timegrid]
    for ind in range(len(timegrid)):
        # we train alpha_t
        base_lr = 0.001
        optimizer = torch.optim.SGD(alphas[ind].parameters(),lr=base_lr, momentum=0.9)
        ind_t = data_batch.data[:,0]==timegrid[ind]
        ind_t = np.where(ind_t.numpy()==1)[0]
        if ind == len(timegrid)-1:
            data_batch_t = terminal_points
        else:
            data_batch_t = data_batch[ind_t,:]
        for epoch in range(epochs):
            lr = base_lr * (0.5 ** (epoch // 5))
            for param_group in optimizer.state_dict()['param_groups']:
                param_group['lr'] = lr
            for row in range(data_batch_t.size()[0]):
                optimizer.zero_grad()
                x = data_batch_t[row]
                xx = data_batch_t[row][1]
                value, _ = value_function(x)
                df_dx = torch.autograd.grad(value, x, create_graph=True)[0]
                df_dx = df_dx[1]
                alpha_x = alphas[ind](xx) 
                loss = c_f*alpha_x**2 + c*alpha_x*df_dx
                print('Time: {:.3f}\t epoch: {}\t row: {}\t loss: {:.3f}'.format(timegrid[ind],epoch,row,loss.data[0]))
                loss.backward()
                optimizer.step()
    return alphas

                
        
    


def iterate(model, criterion, optimizer, losses, time_interval, xlim, batch_size, terminal_size, gamma):
    optimizer.zero_grad()
    data_batch, terminal_points = sample(time_interval, xlim, batch_size, terminal_size)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    
    # the target for the loss function is a vector of zeros, and terminal values        
    target = Variable(torch.zeros(batch_size))
    target_terminal = Variable(gamma*terminal_points.data[:,1]**2)
    
    # the output of the model is solution of pde, and the pde itself for the loss function
    output, PDE = model(data_batch)
    output_terminal, _ = model(terminal_points)
    loss = criterion(PDE, target) #+ criterion(output_terminal, target_terminal)    

    # we get the the gradients of the loss function by backpropagation
    loss.backward()
    # optimization step
    optimizer.step()
    
    losses.update(loss.data[0])

  

def train_LBFGS():
    batch_size = 10
    time_interval = (0,1)
    xlim = (8,10)
    gamma = 2
    model = Model_PDE()
    criterion = nn.MSELoss()
    base_lr = 0.8
    n_iter = 10
    optimizer = torch.optim.LBFGS(model.parameters(),lr=base_lr, max_iter=20)    
    
    # we load all the data
    data_batch, terminal_points = sample(time_interval, xlim, batch_size, gamma)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    
    # the target for the loss function is a vector of zeros, and terminal values        
    target = Variable(torch.zeros(batch_size))
    target_terminal = Variable(gamma*terminal_points.data[:,1]**2)



    for it in range(n_iter):
        
        def closure():
            optimizer.zero_grad()
            # the output of the model is solution of pde, and the pde itself for the loss function
            output, PDE = model(data_batch)
            output_terminal, _ = model(terminal_points)
            loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    
            loss.backward()
            print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))
            return loss
        lr = base_lr * (0.5 ** (it // 5))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
            
        optimizer.step(closure)
    
    # testing
    data_batch, terminal_points = sample(time_interval, xlim, 1, gamma)
    data_batch = Variable(data_batch, requires_grad=True)
    terminal_points = Variable(terminal_points, requires_grad=True)
    output, PDE = model(data_batch)
    output_terminal, _ = model(terminal_points)
    
    # we compute alpha from value function: 
        input_grad = torch.autograd.grad(output[row], data_batch, create_graph=True)[0]
        l.append(input_grad)
    grad_f = reduce(lambda x,y: x+y, l)
    df_dx = grad_f[:,1]
    alpha = -1/(2*model.c_f)*model.c*df_dx
    
    
def train_fixed_iterations():
    batch_size = 20
    n_iter = 5000
    time_interval = (0,10)
    xlim = (8,10)
    gamma = 2
    model = Model_PDE()
    criterion = nn.MSELoss()
    base_lr = 0.001
    optimizer_LBFGS = torch.optim.LBFGS(model.parameters(),lr=base_lr, max_iter=20)  
    optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum=0.9)
    
    # we do an LBFGS to get first guesses for parameters 
    def closure():
        optimizer.zero_grad()
        # the output of the model is solution of pde, and the pde itself for the loss function
        output, PDE = model(data_batch)
        output_terminal, _ = model(terminal_points)
        loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)    
        loss.backward()
        return loss
        
    optimizer_LBFGS.step(closure)    
    
    
    
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    for it in range(n_iter):
        # learning rate decay
        lr = base_lr * (0.5 ** (it // 100))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        # randomly sample t,x
        data_batch, terminal_points = sample(time_interval, xlim, batch_size, gamma)
        data_batch = Variable(data_batch, requires_grad=True)
        terminal_points = Variable(terminal_points, requires_grad=True)
        
        # the target for the loss function is a vector of zeros, and terminal values        
        target = Variable(torch.zeros(batch_size))
        target_terminal = Variable(gamma*terminal_points.data[:,1]**2)
        
        # the output of the model is solution of pde, and the pde itself for the loss function
        output, PDE = model(data_batch)
        output_terminal, _ = model(terminal_points)
        loss = criterion(PDE, target) + criterion(output_terminal, target_terminal)

        # we get the the gradients of the loss function by backpropagation
        loss.backward()
        # optimization step
        optimizer.step()
        
        # print statistics
        print("iteration: [{it}/{n_iter}]\t loss: {loss}".format(it=it, n_iter=n_iter, loss=loss.data[0]))      
            
        
# debugging lines

#i_h1 = nn.Sequential(nn.Linear(2,20,bias=True), nn.Sigmoid())
#h1_h2 = nn.Sequential(nn.Linear(20,20,bias=True), nn.Sigmoid())
#h2_o = nn.Linear(20,1,bias=True)
#
#
#input_all = autograd.Variable(torch.randn(100, 2), requires_grad=True)
#init_time = time.time()
#list_df_dx = []
#list_df_dt = []
#list_df_dxdx = []
#for row in range(input_all.size()[0]):
#    input = input_all[row]
#    h1 = i_h1(input)
#    h2 = h1_h2(h1)
#    output = h2_o(h2)
#    l = []
#    #init_time = time.time()
#    for row in range(output.size()[0]):
#        input_grad = torch.autograd.grad(output[row], input, create_graph=True)[0]
#        l.append(input_grad)
#    end_time = time.time()
#    #print('time gradient: {:.3f}'.format(end_time-init_time))
#    grad_f = reduce(lambda x,y: x+y, l)
#    df_dx = grad_f[1]
#    df_dt = grad_f[0]
#    # we get second derivatives
#    grad_df_dx = []    
#    #init_time = time.time()
#    for row in range(df_dx.size()[0]):
#        grad_df_dx.append(torch.autograd.grad(df_dx[row], input, create_graph = True)[0])
#    end_time = time.time()
#    #print('time hessian: {:.3f}'.format(end_time-init_time))
#    grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
#    df_dxdx = grad_df_dx[1]
#    list_df_dxdx.append(df_dxdx)
#end_time = time.time()
#print('time all: {:.3f}'.format(end_time-init_time))
#df_dxdx_method1 = torch.cat(list_df_dxdx)
#
#
#h1 = i_h1(input_all)
#h2 = h1_h2(h1)
#output = h2_o(h2)
#l = []
##init_time = time.time()
#for row in range(output.size()[0]):
#    input_grad = torch.autograd.grad(output[row], input_all, create_graph=True)[0]
#    l.append(input_grad)
#end_time = time.time()
##print('time gradient: {:.3f}'.format(end_time-init_time))
#grad_f = reduce(lambda x,y: x+y, l)
#df_dx = grad_f[:,1]
#df_dt = grad_f[:,0]
## we get second derivatives
#grad_df_dx = []    
##init_time = time.time()
#for row in range(df_dx.size()[0]):
#    grad_df_dx.append(torch.autograd.grad(df_dx[row], input_all, create_graph = True)[0])
#end_time = time.time()
##print('time hessian: {:.3f}'.format(end_time-init_time))
#grad_df_dx = reduce(lambda x,y: x+y, grad_df_dx)        
#df_dxdx = grad_df_dx[:,1]
#
#
#
#
#
#
#
#
#
#
##output = m(input)
##output = nn.Sigmoid()(output)
##output = nn.Linear(10,1)(output)
##print(output)
##print(output.size())
#
#
#
#
#l = []
#for row in range(output.size()[0]):
#    input_grad = torch.autograd.grad(output[row], input, create_graph=True)[0]
#    l.append(input_grad)
#gradient = reduce(lambda x,y: x+y, l)
#
#gradient_x = gradient[:,1]
#gradient_t = gradient[:,0]
#
#grad_gradient_x = []    
#for batch in range(gradient_x.size()[0]):
#    grad_gradient_x.append(torch.autograd.grad(gradient_x[batch], input, create_graph=True)[0])
#grad_gradient_x = reduce(lambda x,y: x+y, grad_gradient_x)
#
#grad_gradient_t = []    
#for batch in range(gradient_t.size()[0]):
#    grad_gradient_t.append(torch.autograd.grad(gradient_t[batch], input, create_graph=True)[0])
#grad_gradient_t = reduce(lambda x,y: x+y, grad_gradient_t)
#
#
#    
#input_grad = torch.autograd.grad(output, input, create_graph=True)[0]
#hessian = [torch.autograd.grad(input_grad[i], x, create_graph=True)[0] for i in range(x.size()[0])]
#






