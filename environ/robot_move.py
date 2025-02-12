import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2, sqrt
from .model import Model


class RobotMove(Model):

    dt : float = 0.1

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 2 
        self.dim_y = 8 
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0

        self.obs_var = np.ones(self.dim_y) * 0.01
        self.Q = np.eye(self.dim_x) * 0.01
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = np.eye(self.dim_y)

    def f(self, x, u=None):
        dt = self.dt
        x1, x2 = x
        x1_ = x1 + dt
        x2_ = x2 + dt
        return np.array([x1_, x2_])

    def h(self, x):
        landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
        hx = []
        px, py = x
        for i in range(len(landmarks)):
            dist = sqrt((px - landmarks[i][0])**2 + (py - landmarks[i][1])**2)
            angle = np.arctan2(py - landmarks[i][1], px - landmarks[i][0])
            hx.append(dist)
            hx.append(angle)
        return np.array(hx)
        

    def jac_f(self, x, u=0):
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

    # def jac_h(self, x_hat):
    #     return jacobian(lambda x: self.h(x))(x_hat)
    
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
                    cov = 1000 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y, ))