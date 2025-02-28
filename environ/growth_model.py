import numpy as np
from autograd import jacobian
from .model import Model
from .utils import get_beta_mean, get_beta_cov

class GrowthModel(Model):

    dt : float = 0.05

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 3
        self.P0 = np.eye(self.dim_x) * 5
        self.m0 = np.array([5., 5., 5.])
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            self.Q = np.eye(self.dim_x)
            self.R = np.eye(self.dim_y)

        elif noise_type == 'Beta':       
            self.f_alpha = 2.0
            self.f_beta = 2.0
            self.h_alpha = 2.0
            self.h_beta = 2.0

            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x) 
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y) 
        
        elif noise_type == 'Laplace':
            self.f_scale = 1.0
            self.h_scale = 1.0
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        
        else:
            raise NotImplementedError

    def f(self, x, u=None):
        x1, x2, x3 = x
        if u is None:
            x1_ =  x1 / 2 + 25 * x1 / (1+x1**2)
            x2_ = x2 / 3 + 30 * x2 / (1+x2**2)
            x3_ = x3 / 4 + 40 * x3 / (1+x3**2)
        else:
            x1_ =  (x1+0.1*x2) / 2 + 25 * x1 / (1+x1**2 + 0.3*x2**2) + u
            x2_ = (x2+0.1*x3) / 3 + 30 * x2 / (1+x2**2+ 0.5*x3**2) + u
            x3_ = (0.1*x3 + x1) / 4 + 35 * x3 / (1+x3**2+ 0.7*x1**2) + u
            # x_ =  x / 2 + 25 * x / (1+x**2) + u
        return np.array([x1_, x2_, x3_])

    def h(self, x):
        # y = x**2 / 20
        x1, x2, x3 = x
        # y = (x1**2 + x2**2 + x3**2) / 20
        y1 = (x1**2 + x2**2) / 20
        y2 = (x2**2 + x3**2) / 20
        y3 = (x1**2 + x3**2) / 20
        return np.array([y1, y2, y3])
    
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

    def jac_h(self, x):
        x1, x2, x3 = x
        return np.array([
            [x1/10, x2/10, 0],
            [0, x2/10, x3/10],
            [x1/10, 0, x3/10]
        ])