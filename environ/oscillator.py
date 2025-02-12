import autograd.numpy as np
from .model import Model


class Oscillator(Model):

    dt : float = 0.1
    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.F = np.eye(2) + self.dt*np.array([[-0.1, 2],
                                        [-2, -0.1]])
        self.H = np.array([[1, 0],
                           [0, 1]])

        self.dim_x = self.F.shape[0]
        self.dim_y = self.H.shape[0]
        
        self.m0 = np.array([2.5, -5.])
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)

        self.Q = np.array([
            [0.5, 0],
            [0, 0.5],
        ])

        self.noise_type = noise_type
        
        if noise_type == 'Gaussian':
            obs_var = np.array([1., 1.])
            self.R = np.diag(obs_var)

        elif noise_type == 'Beta':       
            self.alpha = 2.0
            self.beta = 5.0
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        
        elif noise_type == 'Laplace':
            self.scale=1
            self.R = self.scale * np.eye(np.dim_y)
        
        else:
            raise ValueError

    def f(self, x, u=None):
        return self.F @ x

    def h(self, x):
        return self.H @ x

    def jac_f(self, x, u=None):
        return self.F

    def jac_h(self, x):
        return self.H

    def f_withnoise(self, x):
        return self.f(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=self.R)
        
        elif self.noise_type == 'Beta':
            raise NotImplementedError
            # noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            # noise = noise - np.mean(noise)
            # return self.h(x) + noise
        
        elif self.noise_type == 'Laplace':
            return self.h(x) + np.random.laplace(loc=0, scale=self.scale, size=(self.dim_y, ))