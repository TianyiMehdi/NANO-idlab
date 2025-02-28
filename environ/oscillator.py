import numpy as np
from .model import Model
from .utils import get_beta_mean, get_beta_cov


class Oscillator(Model):

    dt : float = 0.1
    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.F = np.eye(2) + self.dt*np.array([[-0.1, 2],
                                        [-2, -0.1]])
        self.H = np.array([[1, 1],
                           [-0.5, 1]])

        self.dim_x = self.F.shape[0]
        self.dim_y = self.H.shape[0]
        
        self.m0 = np.array([2.5, -5.])
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)

        self.noise_type = noise_type
        
        if noise_type == 'Gaussian':
            self.Q = 0 * np.eye(self.dim_x)
            self.R = np.eye(self.dim_y)

        elif noise_type == 'Beta':       
            self.f_alpha = 1.5
            self.f_beta = 2.0
            self.h_alpha = 2.0
            self.h_beta = 5.0

            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x) 
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y) 
        
        elif noise_type == 'Laplace':
            self.f_scale = 0.5
            self.h_scale = 1.0
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        
        else:
            raise NotImplementedError

    def f(self, x, u=None):
        return self.F @ x

    def h(self, x):
        return self.H @ x

    def jac_f(self, x, u=None):
        return self.F

    def jac_h(self, x):
        return self.H
    
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
        
        else:
            raise NotImplementedError
    
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
        
        else:
            raise NotImplementedError