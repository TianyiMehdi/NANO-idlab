import numpy as np
from numpy import sin, cos
from .model import Model
from .utils import get_beta_mean, get_beta_cov


class Localization(Model):

    dt : float = 0.1

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 3 
        self.dim_y = 6 
        self.P0 = np.eye(self.dim_x)
        self.m0 = np.zeros(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            self.Q = 0.01 * np.eye(self.dim_x) 
            self.R = 0.01 * np.eye(self.dim_y)

        elif noise_type == 'Beta':       
            self.f_alpha = 4.0
            self.f_beta = 6.0
            self.h_alpha = 4.0
            self.h_beta = 6.0

            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x) 
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y) 
        
        elif noise_type == 'Laplace':
            self.f_scale = 0.01
            self.h_scale = 0.01
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        
        else:
            raise NotImplementedError

    def f(self, x, u=None):
        dt = self.dt
        px, py, phi = x
        if u is None:
            px = px + dt
            py = py + dt
        else:
            v, omega = u
            px = px + v * cos(phi) * dt
            py = py + v * sin(phi) * dt
            phi = phi + omega * dt
        return np.array([px, py, phi])

    def h(self, x):
        # landmarks = np.array([[-1, 2], [-1, 10], [5, 1], [5, 10]])
        landmarks = np.array([[-1, 10], [5, 1], [5, 10]])
        px, py, phi = x
        pos = np.array([px, py])
        # print(phi)
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])
        hx = []
        for i in range(len(landmarks)):
            hx.append(rot.T @ (pos - landmarks[i]))
        y = np.array(hx)
        return y.flatten()
        
    def jac_f(self, x, u=None):
        if u is None:
            return np.eye(3)
        else:
            px, py, phi = x
            v, w = u
            dt = self.dt
            return np.array([
                [1, 0, -v*sin(phi)*dt],
                [0, 1, v*cos(phi)*dt],
                [0, 0, 1]
            ])
    
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