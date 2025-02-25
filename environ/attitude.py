import autograd.numpy as np
from autograd.numpy import sin, cos
from .model import Model
from .utils import (get_beta_mean, get_beta_cov, Omega,
                    quat_mul, quat_Exp, quat_to_rotmat,
                    mtx_w_to_euler_dot, euler_to_rot)

class Attitude(Model):

    dt : float = 0.01

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 3 
        self.dim_y = 6 
        self.P0 = 1e-3*np.eye(self.dim_x)
        self.m0 = np.array([0., 0., 0.])
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)
        # self.x0 = self.x0 / np.linalg.norm(self.x0)
        
        self.noise_type = noise_type
        # self.gyro_std = 5/180*np.pi
        self.g = np.array([0, 0, -9.82])
        self.b = np.array([27.75, -3.65, 47.21])
        # self.b = np.array([0.33, 0, -0.95])
        # self.r1 = np.array([1, 0, 0])
        # self.r2 = np.array([0, 1, 0])

        if noise_type == 'Gaussian':
            # 1e-4 1e-2
            self.Q = 1e-5 * np.eye(self.dim_x) 
            self.R = 1e-4 * np.eye(self.dim_y)

        elif noise_type == 'Beta':       
            self.f_alpha = 5.0
            self.f_beta = 100.0
            self.h_alpha = 5.0
            self.h_beta = 30.0

            # self.Q = 0.03 * np.eye(self.dim_x)
            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x) 
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y) 
        
        elif noise_type == 'Laplace':
            self.f_scale = 1e-3
            self.h_scale = 1e-3
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        
        else:
            raise ValueError

    def f(self, x, u=None):
        dt = self.dt
        deuler = mtx_w_to_euler_dot(x) @ u
        x_ = x + deuler * dt
        return x_

    def h(self, x):
        # rot = quat_to_rotmat(x)
        rot = euler_to_rot(x)
        hx = []
        hx.append(rot.T @ self.g)
        hx.append(rot.T @ self.b)
        y = np.array(hx)
        return y.flatten()
        
    def jac_f(self, x, u=None, epsilon=5e-5):
        """
        使用差分法计算向量值函数的 Jacobian 矩阵
        :param x: 输入向量
        :param epsilon: 差分步长
        :return: Jacobian 矩阵 (m x n)
        """
        n = len(x)  # 输入向量的维度
        f = self.f  # 假设 loss_func 返回的是一个向量
        m = len(f(x, u))  # 输出向量的维度
        jacobian = np.zeros((m, n))  # 初始化 Jacobian 矩阵 (m x n)
        
        fx = f(x, u)  # 计算原始函数值
        
        for i in range(n):
            x_i = x.copy()
            x_i[i] += epsilon  # 对第 i 个元素增加 epsilon
            fx_i = f(x_i, u)  # 计算 perturbed 输出
            
            # 计算 Jacobian 矩阵的每一列
            for j in range(m):
                jacobian[j, i] = (fx_i[j] - fx[j]) / epsilon
        
        return jacobian

    def jac_h(self, x, epsilon=5e-5):
        """
        使用差分法计算向量值函数的 Jacobian 矩阵
        :param x: 输入向量
        :param epsilon: 差分步长
        :return: Jacobian 矩阵 (m x n)
        """
        n = len(x)  # 输入向量的维度
        f = self.h  # 假设 loss_func 返回的是一个向量
        m = len(f(x))  # 输出向量的维度
        jacobian = np.zeros((m, n))  # 初始化 Jacobian 矩阵 (m x n)
        
        fx = f(x)  # 计算原始函数值
        
        for i in range(n):
            x_i = x.copy()
            x_i[i] += epsilon  # 对第 i 个元素增加 epsilon
            fx_i = f(x_i)  # 计算 perturbed 输出
            
            # 计算 Jacobian 矩阵的每一列
            for j in range(m):
                jacobian[j, i] = (fx_i[j] - fx[j]) / epsilon
        
        return jacobian
    
    def f_withnoise(self, x, u=None):
        if self.noise_type == 'Gaussian':
            x_noise = self.f(x, u) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
            # print(np.linalg.norm(x_noise))
            # x_noise = x_noise / np.linalg.norm(x_noise)
            return x_noise
        
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.f_alpha, self.f_beta, self.dim_x)
            mean = get_beta_mean(self.f_alpha, self.f_beta)
            noise = noise - mean
            return self.f(x, u) + noise
        
        elif self.noise_type == 'Laplace':
            return self.f(x, u) + np.random.laplace(loc=0, scale=self.f_scale, size=(self.dim_x, ))

        else:
            raise ValueError
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            prob = np.random.rand()
            if prob <= 0.9:
                cov = self.R  # 95%概率使用R
            else:
                cov = 1000 * self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.h_alpha, self.h_beta, self.dim_y)
            mean = get_beta_mean(self.h_alpha, self.h_beta)
            noise = noise - mean
            return self.h(x) + noise
        
        elif self.noise_type == 'Laplace':
            return self.h(x) + np.random.laplace(loc=0, scale=self.h_scale, size=(self.dim_y, ))
        
        else:
            raise ValueError