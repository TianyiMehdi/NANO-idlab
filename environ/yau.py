from dataclasses import dataclass
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from .model import Model

class Yau(Model):

    dt : float = 1.0

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 2
        self.dim_y = 2
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)
        
        self.Q = np.eye(self.dim_x) * 0.1

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            obs_var = np.ones(self.dim_y) * 0.1
            self.R = np.diag(obs_var)

        elif noise_type == 'Beta':       
            self.alpha = 2.0
            self.beta = 5.0
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        
        elif noise_type == 'Laplace':
            self.scale=1
            self.R = self.scale * np.eye(self.dim_y)
        
        else:
            raise ValueError

    def f(self, x, u=None):
        return x + cos(x)

    def h(self, x):
        y = x**3
        return y
    
    def f_withnoise(self, x, u=None):
        return self.f(x, u) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=self.R)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            mean = self.alpha / (self.alpha + self.beta)
            noise = noise - mean
            return self.h(x) + noise
        elif self.noise_type == 'Laplace':
            return self.h(x) + np.random.laplace(loc=0, scale=self.scale, size=(self.dim_y,))
        else:
            raise ValueError
    
    def jac_f(self, x_hat, u=None):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x):
        return np.array([
            [3*x[0]**2, 0],
            [0, 3*x[1]**2]
        ])