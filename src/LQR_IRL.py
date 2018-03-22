import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.integrate import trapz, simps
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
#PATH_IMAGES = '/Users/msabate/Projects/Turing/Mean-Field-Games/images' # save this



class Policy_Iteration_grad():    
    
    def __init__(self, x_0=0, b=0.5, c=0.5, sigma=1, b_f=0.5, c_f=0.9, gamma=1, T=10, init_t = 9, solver='Euler', timestep=0.05):
        self.x_0 = x_0
        self.b = b
        self.c = c
        self.sigma = sigma
        self.b_f = b_f
        self.c_f = c_f
        self.gamma = gamma
        self.init_t = init_t
        self.T = T
        self.n = 1 # step of iteration. VERY IMPORTANT
        self.timestep = 0.05
        self.time = np.arange(self.init_t,self.T+0.1*timestep,timestep) # time is discretised in 1000 points
        self.init_p1 = None 
        self.init_p2 = None 
        self.p1_grid, self.p2_grid = [], []
        self.init_p()
        self.beta, self.phi, self.delta = [], [], [] # this will store the functions beta(t), phi(t),
        self.solver = solver
        self.grad_beta, self.grad_delta, self.grad_phi = None, None, None
        self.get_grad = False
        
    def init_p(self):
        """
        Alpha_0 needs to be linear in x
        """
        #self.p1 = self.t  
        self.init_p1= lambda t: -0.5
        self.init_p2 = lambda t: 0.1
        self.p1_grid.append(np.array([self.init_p1(t) for t in self.time]))
        self.p2_grid.append(np.array([self.init_p2(t) for t in self.time]))
    
    def get_alpha(self,x):
        """
        get alpha^n(t,x), where n is the last step of the iteration
        """
        res = x*self.p1_grid[-1] + self.p2_grid[-1]
        return res  # a vector of length time
    
    def get_value_function(self, x):
        beta_array = np.array(self.beta[-1])
        delta_array = np.array(self.delta[-1])
        phi_array = np.array(self.phi[-1])
        res = (x**2)*beta_array + x*delta_array + phi_array
        return res
    
    def get_grad_value_funciton(self,x):
        res_grad_bf = (x**2)*self.grad_beta[0] + x*self.grad_delta[0] + self.grad_phi[0]
        res_grad_cf = (x**2)*self.grad_beta[1] + x*self.grad_delta[1] + self.grad_phi[1]
        res_grad_gamma = (x**2)*self.grad_beta[2] + x*self.grad_delta[2] + self.grad_phi[2]
        return res_grad_bf,res_grad_cf,res_grad_gamma
        
     

    def evaluation_step(self):
        """
        This function solves the PDE given by HJB at the current iteration
        To solve a PDE we make a guess w of the solution, we take coefficients of
        x^2, x and 1, and create a system of 3 ODEs. 
        we create the system of three equations below
        
        We update the values of p1 and p2, that will be used in the next iteration
        
        Output:
            solution of the PDE
        """
        current_step = self.n
        # first ode: d beta(t) = (beta0(t) + beta1(t)beta(t))dt
        beta0 = [-(self.b_f + self.c_f*self.p1_grid[current_step-1][t]**2) for t in range(len(self.time))]
        beta1 = [-(2*self.b + 2*self.c*self.p1_grid[current_step-1][t]) for t in range(len(self.time))]
        if self.solver=='Euler':
            self.beta.append(self._solve_ode_euler(beta0, beta1, self.gamma)) # beta is a funcation lambda
        else:
            self.beta.append(self._solve_ode_explicit(beta0, beta1, self.gamma)) # beta is a funcation lambda
        
        # second ode: d delta(t) = (delta0(t) + delta1(t)delta(t))dt
        delta0 = [-(2*self.c_f * self.p1_grid[current_step-1][t] * self.p2_grid[current_step-1][t] + 2*self.c*self.beta[current_step-1][t]*self.p2_grid[current_step-1][t]) for t in range(len(self.time))]
        delta1 = [-(self.b + self.c*self.p1_grid[current_step-1][t]) for t in range(len(self.time))]
        if self.solver == 'Euler':
            self.delta.append(self._solve_ode_euler(delta0, delta1, 0)) # delta is a function lambda
        else:
            self.delta.append(self._solve_ode_explicit(delta0, delta1, 0)) # delta is a function lambda
            
        # third ode: d phi = (phi0(t) + phi1(t)phi(t))dt
        phi0 =  [-(self.sigma**2*self.beta[current_step-1][t] + self.c_f*self.p2_grid[current_step-1][t]**2 + self.c*self.delta[current_step-1][t]*self.p2_grid[current_step-1][t]) for t in range(len(self.time))]
        phi1 = [0]*len(self.time)
        if self.solver == 'Euler':
            self.phi.append(self._solve_ode_euler(phi0, phi1, 0)) # phi is a function lambda`A
        else:
            self.phi.append(self._solve_ode_explicit(phi0, phi1, 0)) # phi is a function lambda`A
        
        # we get gradient of beta, delta and phi for IRL
        if self.get_grad:
            self._grad_beta(beta0, beta1)
            self._grad_delta(delta0, delta1)
            self._grad_phi(phi0, phi1)
        
        # we update p1 and p2:
        p1_new = np.array([-self.c/(2*self.c_f)*2*self.beta[current_step-1][t] for t in range(len(self.time))])
        p2_new = np.array([-self.c/(2*self.c_f)*self.delta[current_step-1][t] for t in range(len(self.time))])
        self.p1_grid.append(p1_new)
        self.p2_grid.append(p2_new)
        self.n += 1
        
        
    def _solve_ode_euler(self, a0, a1, y_T):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = y_T
        Input:
            - a0: grid of points between [0,T]
            - a1: grid of points between [0,T]
            - y_T = y(T)
        
        Output:
            - solution of ODE evaluated at the grid of time points beteen [0,T]
        Note: We apply Euler method to solve this ODE 
        """
        # y_T is terminal condition
        y0 = y_T
        res = [0]*len(self.time)
        res[-1] = y0
        for t in range(len(self.time)-1, 0, -1):
            m = a0[t]+a1[t]*res[t]
            y1 = y0 - m*(self.time[t]-self.time[t-1])
            res[t-1] = y1
            y0 = y1
        return res
    
    def _solve_ode_explicit(self, a0, a1, y_T):
        """
        This function solves a specific type of ODE: dy(t)=(a0(t) + a1(t)*y(t))dt; y(T) = gamma
        The solution is:
            y(s) = y(T)*exp[-int_s^T a1(t)dt] - int_s^T[a0(t)exp[-int_s^t a1(r)dr]]dt
        
        Input:
            - a0: function
            - a1: function
            - y_T: value of y(T)
        
        Output:
            - solution of ODE at time grid points. 
            
        Note: we use Simpson's rule to solve the integrals
        Note: This ffunction still doesn't work. Need to be corrected
        """
        a0 = np.array(a0)
        a1 = np.array(a1)
        integrand_const = -a1
        val_integral_const = simps(integrand_const, self.time)
        integrand2 = np.zeros_like(self.time)
        val_integral2 = np.zeros_like(self.time)
        val_integral3 = np.zeros_like(self.time)
        val_integral4 = np.zeros_like(self.time)
        
        # we fill integrand4
        for i in range(1,len(self.time)):
            integrand4 = -np.copy(a1[i-1:i+1])
            val = simps(integrand4, self.time[i-1:i+1])
            val_integral4[i] = val
        val_integral4 = np.cumsum(val_integral4)
        e_integral4 = np.exp(-val_integral4)
        
        # we fill val integral3
        for i in range(1,len(self.time)):
            integrand3 = -np.copy(a1[i-1:i+1])
            val = simps(integrand3, self.time[i-1:i+1])
            val_integral3[i] = val
        val_integral3 = np.cumsum(val_integral3)
        
        # we get integrand2
        integrand2 = a0 * np.exp(val_integral3)
        
        # we get val_integral2
        for i in range(len(self.time)-1,0,-1):
            val = simps(integrand2[i-1:i+1], self.time[i-1:i+1])
            val_integral2[i-1] = val
        
        val_integral2 = np.flip(np.cumsum(np.flip(val_integral2,axis=0)),axis=0)
        
        solution_ode = e_integral4*(y_T*math.exp(val_integral_const)-val_integral2)
        return(solution_ode)
    
    def _grad_beta(self,a0,a1):
        # grad_b_f
        a0 = np.array(a0)
        a1 = np.array(a1)
        val_integral4 = np.zeros_like(self.time)
        val_integral3 = np.zeros_like(self.time)
        val_integral2_bf = np.zeros_like(self.time)
        val_integral2_cf = np.zeros_like(self.time)
        
        # we fill integrand4
        for i in range(1,len(self.time)):
            integrand4 = -np.copy(a1[i-1:i+1])
            val = trapz(integrand4, self.time[i-1:i+1])
            val_integral4[i] = val
        val_integral4 = np.cumsum(val_integral4)
        e_integral4 = np.exp(-val_integral4)
        
        # we fill val integral3
        for i in range(1,len(self.time)):
            integrand3 = -np.copy(a1[i-1:i+1])
            val = trapz(integrand3, self.time[i-1:i+1])
            val_integral3[i] = val
        val_integral3 = np.cumsum(val_integral3)
        
        # we fill integrand2
        integrand2_bf = np.exp(val_integral3)
        
        # we get val_integral2
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_bf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_bf[i-1] = val
        
        val_integral2_bf = np.flip(np.cumsum(np.flip(val_integral2_bf,axis=0)),axis=0)
        
        # we get val_integral2_cf
        integrand2_cf = self.p1_grid[-1]**2 * np.exp(val_integral3)
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_cf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_cf[i-1] = val
        
        val_integral2_cf = np.flip(np.cumsum(np.flip(val_integral2_cf,axis=0)),axis=0)
        
        grad_bf = e_integral4*val_integral2_bf
        grad_cf = e_integral4*val_integral2_cf
        
        integrand_const = -a1
        val_integral_const = simps(integrand_const, self.time)
        grad_gamma = e_integral4*math.exp(val_integral_const)
        
        self.grad_beta = (grad_bf, grad_cf, grad_gamma)
    
    def _grad_delta(self, a0, a1):
        a0 = np.array(a0)
        a1 = np.array(a1)
        val_integral4 = np.zeros_like(self.time)
        val_integral3 = np.zeros_like(self.time)
        val_integral2_bf = np.zeros_like(self.time)
        val_integral2_cf = np.zeros_like(self.time)
        val_integral2_gamma = np.zeros_like(self.time)
        
        # we fill integrand4
        for i in range(1,len(self.time)):
            integrand4 = -np.copy(a1[i-1:i+1])
            val = trapz(integrand4, self.time[i-1:i+1])
            val_integral4[i] = val
        val_integral4 = np.cumsum(val_integral4)
        e_integral4 = np.exp(-val_integral4)
        
        # we fill integrand3
        for i in range(1,len(self.time)):
            integrand3 = -np.copy(a1[i-1:i+1])
            val = trapz(integrand3, self.time[i-1:i+1])
            val_integral3[i] = val
        val_integral3 = np.cumsum(val_integral3)
        
        # we fill integrand2
        integrand2_bf = 2*self.c*self.grad_beta[0]*self.p2_grid[-1]*np.exp(val_integral3)
        # we get val_integral2
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_bf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_bf[i-1] = val
        
        val_integral2_bf = np.flip(np.cumsum(np.flip(val_integral2_bf,axis=0)),axis=0)

        integrand2_cf = 2*self.p1_grid[-1]*self.p2_grid[-1]* + 2*self.c*self.grad_beta[1]*self.p2_grid[-1]*np.exp(val_integral3)

        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_cf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_cf[i-1] = val
        
        val_integral2_cf = np.flip(np.cumsum(np.flip(val_integral2_cf,axis=0)),axis=0)
        
        integrand2_gamma = 2*self.c*self.grad_beta[2]*self.p2_grid[-1]*np.exp(val_integral3)
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_gamma[i-1:i+1], self.time[i-1:i+1])
            val_integral2_gamma[i-1] = val
        
        val_integral2_gamma = np.flip(np.cumsum(np.flip(val_integral2_gamma,axis=0)),axis=0)
        
        grad_bf = e_integral4*val_integral2_bf
        grad_cf = e_integral4*val_integral2_cf
        grad_gamma = e_integral4*val_integral2_gamma
        
        self.grad_delta = (grad_bf, grad_cf, grad_gamma)
        
    def _grad_phi(self,a0,a1):
        a0 = np.array(a0)
        a1 = np.array(a1)
        #val_integral4 = np.zeros_like(self.time)
        #val_integral3 = np.zeros_like(self.time)
        val_integral2_bf = np.zeros_like(self.time)
        val_integral2_cf = np.zeros_like(self.time)
        val_integral2_gamma = np.zeros_like(self.time)
        
        integrand2_bf = self.sigma**2 * self.grad_beta[0] + self.c*self.grad_delta[0]*self.p2_grid[-1]
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_bf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_bf[i-1] = val
        
        val_integral2_bf = np.flip(np.cumsum(np.flip(val_integral2_bf,axis=0)),axis=0)
        
        integrand2_cf = self.sigma**2 * self.grad_beta[1] + self.p2_grid[-1]**2 + self.c * self.grad_delta[1]*self.p2_grid[-1]

        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_cf[i-1:i+1], self.time[i-1:i+1])
            val_integral2_cf[i-1] = val
        
        val_integral2_cf = np.flip(np.cumsum(np.flip(val_integral2_cf,axis=0)),axis=0)
        
        integrand2_gamma = self.sigma**2*self.grad_beta[2] + self.c*self.grad_delta[2]*self.p2_grid[-1]
        for i in range(len(self.time)-1,0,-1):
            val = trapz(integrand2_gamma[i-1:i+1], self.time[i-1:i+1])
            val_integral2_gamma[i-1] = val
        
        val_integral2_gamma = np.flip(np.cumsum(np.flip(val_integral2_gamma,axis=0)),axis=0)
        
        self.grad_phi = (val_integral2_bf, val_integral2_cf, val_integral2_gamma)


        

def iterate(x_0, b, c, b_f, c_f, gamma, T, init_t, n_iterations, solver, timestep):

    pol = Policy_Iteration_Euler(x_0=x_0, b=b, c=c, sigma=sigma, b_f=b_f, c_f=c_f, 
                                 gamma=gamma, T=T, init_t =init_t, solver=solver, timestep=timestep)
    #pol = Policy_Iteration_Euler()
    x = np.linspace(0,10,10)
    n_iterations=50
    
    alphas = []
    value_functions = []
    
    alpha = np.array([pol.get_alpha(x_i) for x_i in x]) # initial guess for alpha on the grid of points
    alphas.append(alpha)
    
    for i in range(n_iterations):
        print('iteration = {}'.format(i))
        if i == n_iterations-1:
            pol.get_grad=True
        pol.evaluation_step()
        alpha = np.array([pol.get_alpha(x_i) for x_i in x])
        alphas.append(alpha)
        value = np.array([pol.get_value_function(x_i) for x_i in x])
        value_functions.append(value)
    
    diff_alphas = [norm(alphas[i+1]-alphas[i], ord='fro') for i in range(len(alphas)-1)]
    diff_value = [norm(value_functions[i+1]-value_functions[i], ord='fro') for i in range(len(value_functions)-1)]
    
    return pol, alphas, value_functions, diff_alphas, diff_value
  
def main():
    x_0 = 0
    init_t, T = 0,1
    timestep = 0.05
    xlim = (0,5)
    b=0.5 
    c=0.5 
    sigma=1 
    b_f=0.5 
    c_f=0.9
    gamma = 1
    n_iterations = 50   
    solver = 'explicit'    
    timestep = 0.05
    
    # Target policy
    pol, alphas, value_functions, _, _ = iterate(x_0, b, c, b_f, c_f, gamma, T, init_t, n_iterations, solver, timestep)
    
    # Gradient Descent
    # 1. we start with random values for b_f, c_f, gamma
    b_f = random.gauss(0,1)
    c_f = random.gauus(0,1)
    gamma = random.gauss(0,1)
    
    pol_pred, alphas_pred, value_functions_pred, _, _ = iterate(x_0, b, c, b_f, c_f, gamma, T, init_t, n_iterations, solver, timestep)

    J = norm(value_functions_pred[-1] - value_functions[-1])
        
            
        
        
        
         
        
    