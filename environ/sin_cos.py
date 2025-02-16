import autograd.numpy as np
from autograd import jacobian
from autograd.numpy import sin, cos
from .model import Model
from .utils import get_beta_mean, get_beta_cov


class SinCos(Model):

    dt : float = 1.0

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 2
        self.dim_y = 2
        self.P0 = 25 * np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)
        

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            self.Q = 2**2 * np.eye(self.dim_x)
            self.R = 1 * np.eye(self.dim_y)

        elif noise_type == 'Beta':       
            self.f_alpha = 1.5
            self.f_beta = 2.0
            self.h_alpha = 3.0
            self.h_beta = 7.0

            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x) 
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y) 
        
        elif noise_type == 'Laplace':
            self.f_scale = 2**2
            self.h_scale = 1.0
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        
        else:
            raise ValueError

    def f(self, x, u=None):
        #  parmater
        k1 = 0.1
        k2 = 0.1
        F = np.eye(self.dim_x) + k1 * np.array([[-1,0],[0.1,-1]])
        x_ = F @ x + k2 * cos(x)
        return x_

    def h(self, x):
        y = x + sin(x)
        return y
    
    def f_withnoise(self, x, u=None):
        if self.noise_type == 'Gaussian':
            return self.f(x, u) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
        
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.f_alpha, self.f_beta, self.dim_x)
            mean = get_beta_mean(self.f_alpha, self.f_beta)
            noise = noise - mean
            return self.f(x, u) + noise
        
        elif self.noise_type == 'Laplace':
            return self.f(x, u) + np.random.laplace(loc=0, scale=self.f_scale, size=(self.dim_x, ))
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=self.R)
        
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.h_alpha, self.h_beta, self.dim_y)
            mean = get_beta_mean(self.h_alpha, self.h_beta)
            noise = noise - mean
            return self.h(x) + noise
        
        elif self.noise_type == 'Laplace':
            return self.h(x) + np.random.laplace(loc=0, scale=self.h_scale, size=(self.dim_y, ))
    
    def jac_f(self, x_hat, u=None):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x_hat):
        return np.array([
            [1 + cos(x_hat[0]), 0],
            [0, 1 + cos(x_hat[1])]
        ])