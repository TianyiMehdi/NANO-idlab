from dataclasses import dataclass
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from .model import Model

class SinCos(Model):

    dt : float = 1.0

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 2
        self.dim_y = 2
        self.P0 = 25 * np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)
        
        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0
        self.obs_var = np.ones(self.dim_y)

        self.Q = np.eye(self.dim_x) * 4
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / (
                        (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = np.eye(self.dim_y)
        

    def f(self, x, u=0):
        #  parmater
        k1 = 0.1
        k2 = 0.1
        F = np.eye(self.dim_x) + k1 * np.array([[-1,0],[0.1,-1]])
        x_ = F @ x + k2 * cos(x)
        return x_

    def h(self, x):
        y = x + sin(x)
        return y
    
    def f_withnoise(self, x):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.95:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=cov)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.95:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 10 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y,))

    def jac_f(self, x_hat, u=0):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x_hat, u=0):
        return jacobian(lambda x: self.h(x))(x_hat)