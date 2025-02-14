import autograd.numpy as np
from autograd import jacobian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from .model import Model

class Toy(Model):

    dt : float = 1.0

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 3
        self.P0 = np.eye(self.dim_x) * 5
        self.m0 = np.array([5., 5., 5.])
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)
        
        self.Q = np.eye(self.dim_x)

        self.noise_type = noise_type

        if noise_type == 'Gaussian':
            obs_var = np.ones(self.dim_y)
            self.R = np.diag(obs_var)

        elif noise_type == 'Beta':       
            self.alpha = 2.0
            self.beta = 5.0
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        
        elif noise_type == 'Laplace':
            self.scale = 1
            self.R = self.scale * np.eye(self.dim_y)
        
        else:
            raise ValueError

    def f(self, x, u=None):
        x1, x2, x3 = x
        if u is None:
            x1_ =  x1 / 2 + 25 * x1 / (1+x1**2)
            x2_ = x2 / 3 + 30 * x2 / (1+x2**2)
            x3_ = x3 / 4 + 40 * x3 / (1+x3**2)
        else:
            x1_ =  x1 / 2 + 25 * x1 / (1+x1**2) + u
            x2_ = x2 / 3 + 30 * x2 / (1+x2**2) + u
            x3_ = x3 / 4 + 35 * x3 / (1+x3**2) + u
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
        return jacobian(lambda x: self.f(x, u))(x_hat)
    
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