import autograd.numpy as np
from .model import Model


class Robot(Model):

    dt : float = 0.1

    def __init__(self, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 2 
        self.dim_y = 8 
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)

        self.Q = np.eye(self.dim_x) * 0.01

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
        dt = self.dt
        if u == None:
            x1, x2 = x
            x1_ = x1 + dt
            x2_ = x2 + dt
        else:
            x1, x2 = x
            x1_ = x1 + u * np.cos(np.pi/3) * dt
            x2_ = x2 + u * np.sin(np.pi/3) * dt
        return np.array([x1_, x2_])

    def h(self, x):
        landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
        hx = []
        px, py = x
        for i in range(len(landmarks)):
            dist = np.sqrt((px - landmarks[i][0])**2 + (py - landmarks[i][1])**2)
            angle = np.arctan2(py - landmarks[i][1], px - landmarks[i][0])
            hx.append(dist)
            hx.append(angle)
        return np.array(hx)
        

    def jac_f(self, x, u=None):
        return np.array([[1, 0], [0, 1]])

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
            return self.h(x) + np.random.laplace(loc=0, scale=self.scale, size=(self.dim_y, ))
        else:
            raise ValueError