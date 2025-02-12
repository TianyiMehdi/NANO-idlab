import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from .model import Model

landmarks = np.array([
    [0, 0], [0, 25], [0, 50], [0, 75], [0, 100],
    [25, 0], [25, 25], [25, 50], [25, 75], [25, 100],
    [50, 0], [50, 25], [50, 50], [50, 75], [50, 100],
    [75, 0], [75, 25], [75, 50], [75, 75], [75, 100],
    [100, 0], [100, 25], [100, 50], [100, 75], [100, 100],
])

# landmarks = np.array([
#     [0, 0], [0, 25],
#     [25, 0], [25, 25],
# ])

class Sensor_Network(Model):

    dt : float = 0.01
    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           ])

        self.dim_x = 4
        self.dim_y = 25
        self.P0 = np.diag(np.array([49, 1, 4, 2]))
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0

        self.obs_var = np.full(self.dim_y, 0.1)
        self.Q = 0.04 * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = np.diag(self.obs_var)

    def f(self, x, u=None):
        return self.F @ x

    def h(self, x):
        W0 = 1000
        d0 = 1
        hx = []
        px, py, vx, vy = x
        for i in range(self.dim_y):
            dist = (px - landmarks[i][0])**2 + (py - landmarks[i][1])**2
            # if dist <= d0:
            #     hx.append(np.sqrt(W0))
            # else:
            #     hx.append(np.sqrt(W0*d0/dist))
            hx.append(np.sqrt(dist))

        return np.array(hx)

    def f_withnoise(self, x):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.9:
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
                if prob <= 0.9:
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
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y,))
        
    def jac_f(self, x_hat, u=0):
        return self.F

    # def jac_h(self, x_hat, u=0):
    #     return jacobian(lambda x: self.h(x))(x_hat)
    
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