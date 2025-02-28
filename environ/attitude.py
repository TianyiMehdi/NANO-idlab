import numpy as np
from .model import Model
from .utils import (get_beta_mean, mtx_w_to_euler_dot, euler_to_rot)

class Attitude(Model):

    dt : float = 0.01

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 3 
        self.dim_y = 6 
        self.P0 = 1e-3*np.eye(self.dim_x)
        self.m0 = np.array([0., 0., 0.])
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)
        
        self.noise_type = noise_type
        self.g = np.array([0, 0, -9.82])
        self.b = np.array([27.75, -3.65, 47.21])

        # Parameters for Laplace noise
        self.laplace_scale = 1e-5
        
        # Parameters for Beta noise
        self.beta_alpha = 1.2
        self.beta_beta = 1.5
        
        # Parameters for Filter covariance
        self.Q = self.laplace_scale * np.eye(self.dim_x) 
        self.R = 1e-4 * np.eye(self.dim_y)
        
        self.trans_pc = 0.1
        self.mea_pc = 0.15

    def f(self, x, u=None):
        dt = self.dt
        deuler = mtx_w_to_euler_dot(x) @ u
        x_ = x + deuler * dt
        return x_

    def h(self, x):
        rot = euler_to_rot(x)
        hx = []
        hx.append(rot.T @ self.g)
        hx.append(rot.T @ self.b)
        y = np.array(hx)
        return y.flatten()
          
    def f_withnoise(self, x, u=None):
        prob = np.random.rand()
        if prob <= 1-self.trans_pc:
            scale = self.laplace_scale
        else:
            scale = 1000 * self.laplace_scale
        return self.f(x, u) + np.random.laplace(loc=0, scale=scale, size=(self.dim_x, ))
    
    def h_withnoise(self, x):
        prob = np.random.rand()
        if prob <= 1-self.mea_pc:
            cov = self.R 
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        else:
            noise = np.random.beta(self.beta_alpha, self.beta_beta, self.dim_y)
            mean = get_beta_mean(self.beta_alpha, self.beta_beta)
            noise = noise - mean
            return self.h(x) + noise
        