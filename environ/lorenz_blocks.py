import numpy as np

from .model import Model
from .utils import get_beta_cov, get_beta_mean


class LorenzBlocks(Model):
    """Block-diagonal extension of Lorenz-63 dynamics."""

    dt: float = 0.01

    def __init__(self, noise_type="Gaussian", num_blocks=3, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        super().__init__(self)

        self.num_blocks = num_blocks
        self.dim_x = 3 * num_blocks
        self.dim_y = self.dim_x

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.m0 = np.zeros(self.dim_x)
        self.P0 = 0.1 * np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=self.m0, cov=self.P0)

        self.noise_type = noise_type

        if noise_type == "Gaussian":
            self.Q = 0.05 * np.eye(self.dim_x)
            self.R = 0.04 * np.eye(self.dim_y)
        elif noise_type == "Beta":
            self.f_alpha = 2.0
            self.f_beta = 5.0
            self.h_alpha = 2.0
            self.h_beta = 5.0
            self.Q = get_beta_cov(self.f_alpha, self.f_beta) * np.eye(self.dim_x)
            self.R = get_beta_cov(self.h_alpha, self.h_beta) * np.eye(self.dim_y)
        elif noise_type == "Laplace":
            self.f_scale = 0.05
            self.h_scale = 0.1
            self.Q = self.f_scale * np.eye(self.dim_x)
            self.R = self.h_scale * np.eye(self.dim_y)
        else:
            raise NotImplementedError

    def _reshape(self, x):
        return np.asarray(x).reshape(self.num_blocks, 3)

    def f(self, x, u=None):
        blocks = self._reshape(x)
        drift = np.zeros_like(blocks)
        for i, (x1, x2, x3) in enumerate(blocks):
            dx = self.sigma * (x2 - x1)
            dy = x1 * (self.rho - x3) - x2
            dz = x1 * x2 - self.beta * x3
            drift[i] = np.array([dx, dy, dz])

        if u is not None:
            drift += self._reshape(u)

        return (blocks + self.dt * drift).reshape(-1)

    def h(self, x):
        blocks = self._reshape(x)
        y = np.zeros_like(blocks)
        y[:, 0] = np.sin(0.5 * blocks[:, 0])
        y[:, 1] = np.tanh(blocks[:, 1])
        y[:, 2] = np.exp(-0.1 * blocks[:, 2])
        return y.reshape(-1)

    def jac_f(self, x, u=None):
        blocks = self._reshape(x)
        jac = np.zeros((self.dim_x, self.dim_x))
        for i, (x1, x2, x3) in enumerate(blocks):
            block = np.array(
                [
                    [-self.sigma, self.sigma, 0.0],
                    [self.rho - x3, -1.0, -x1],
                    [x2, x1, -self.beta],
                ]
            )
            idx = slice(3 * i, 3 * (i + 1))
            jac[idx, idx] = block
        return np.eye(self.dim_x) + self.dt * jac

    def jac_h(self, x):
        blocks = self._reshape(x)
        jac = np.zeros((self.dim_y, self.dim_x))
        for i, (x1, x2, x3) in enumerate(blocks):
            idx = slice(3 * i, 3 * (i + 1))
            jac[idx, idx] = np.diag(
                [
                    0.5 * np.cos(0.5 * x1),
                    1.0 - np.tanh(x2) ** 2,
                    -0.1 * np.exp(-0.1 * x3),
                ]
            )
        return jac

    def f_withnoise(self, x, u=None):
        if self.noise_type == "Gaussian":
            return self.f(x, u) + np.random.multivariate_normal(np.zeros(self.dim_x), self.Q)
        if self.noise_type == "Beta":
            noise = np.random.beta(self.f_alpha, self.f_beta, self.dim_x)
            return self.f(x, u) + (noise - get_beta_mean(self.f_alpha, self.f_beta))
        if self.noise_type == "Laplace":
            return self.f(x, u) + np.random.laplace(loc=0.0, scale=self.f_scale, size=self.dim_x)
        raise NotImplementedError

    def h_withnoise(self, x):
        if self.noise_type == "Gaussian":
            return self.h(x) + np.random.multivariate_normal(np.zeros(self.dim_y), self.R)
        if self.noise_type == "Beta":
            noise = np.random.beta(self.h_alpha, self.h_beta, self.dim_y)
            return self.h(x) + (noise - get_beta_mean(self.h_alpha, self.h_beta))
        if self.noise_type == "Laplace":
            return self.h(x) + np.random.laplace(loc=0.0, scale=self.h_scale, size=self.dim_y)
        raise NotImplementedError
