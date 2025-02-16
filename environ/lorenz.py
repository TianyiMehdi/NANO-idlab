import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from scipy.linalg import expm
from .model import Model

class Lorenz(Model):

    dt : float = 0.1

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 5
        self.m0 = 1 * np.ones(self.dim_x)
        self.P0 = 0.4 * np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)
        
        self.Q = np.eye(self.dim_x) * 1e-4

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            obs_var = np.ones(self.dim_y) * 1e-6
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
        #  parmater
        A_x = np.array([[-10, 10, 0],
                        [28, -1, -x[0]],  
                        [0, x[0], -8/3]    
                        ])
        return expm(A_x * self.dt) @ x

    def h(self, x):
        rho = np.sqrt(np.sum(np.square(x)))
        r = np.sqrt(np.sum(np.square(x[:2])))
        cos_theta = x[0] / r
        sin_theta = x[1] / r
        cos_phi = r / rho
        sin_phi = x[2] / rho
        return np.array([rho, cos_theta, sin_theta, cos_phi, sin_phi])
    
    def f_withnoise(self, x, u=None):
        return self.f(x, u) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=self.R)
        elif self.noise_type == 'Beta':
            # noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            # noise = noise - np.mean(noise)
            # return self.h(x) + noise
            raise NotImplementedError
        elif self.noise_type == 'Laplace':
            return self.h(x) + np.random.laplace(loc=0, scale=self.scale, size=(self.dim_y,))
        else:
            raise ValueError
    
    def jac_f(self, x, u=None, epsilon=5e-5):
        """
        使用差分法计算向量值函数的 Jacobian 矩阵
        :param x: 输入向量
        :param epsilon: 差分步长
        :return: Jacobian 矩阵 (m x n)
        """
        n = len(x)  # 输入向量的维度
        f = self.f  # 假设 loss_func 返回的是一个向量
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
    
    def jac_h(self, x, epsilon=1e-7):
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