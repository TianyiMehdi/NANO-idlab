import autograd.numpy as np
from autograd import hessian
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from filterpy.kalman import unscented_transform as UT
from scipy.optimize import minimize

from .utils import (cal_mean, is_positive_semidefinite,
                    kl_divergence)

class NANO:
    threshold: float = 1e-4

    def __init__(self, model, **filter_dict):    
        self.model = model
        self.dim_x = model.dim_x
        self.dim_y = model.dim_y    
        self.x = model.x0
        self.P = model.P0

        self.f = model.f
        self.h = model.h
        self.jac_f = model.jac_f
        self.jac_h = model.jac_h
        self.Q = model.Q
        self.R = model.R
        self._I = np.eye(self.dim_x)

        self.n_iterations = filter_dict['n_iterations']
        self.points = JulierSigmaPoints(self.dim_x, kappa=0)
        # self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.45, beta=2.0, kappa=1.0)
        # self.sigmas_f = np.zeros((self.points.num_sigmas, self.dim_x))
        self.x_prior = model.x0
        self.P_prior = model.P0
        self.x_post = model.x0
        self.P_post = model.P0
        
        self.init_type = filter_dict['init_type']
        if self.init_type == 'iekf':
            self.iekf_max_iter = filter_dict['iekf_max_iter']
        self.lr = filter_dict['lr']

        self.loss_func = self.log_likelihood_loss
    
    # 可以改成 @    写loss可以写个引用的文章
    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
     
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.loss_func(x_posterior, y)
        return l1 + l2
            
    def update_init(self, y, x_prior, P_prior):
        # Laplace Approximation
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        return x_hat_posterior, P_posterior_inv
    
    def update_iekf_init(self, y, x_prior, P_prior, max_iter=1):
        # Iterated Extended Kalman Filter (IEKF) for Maximum A Posteriori (MAP)
        x_hat = x_prior
        for i in range(max_iter):
            H = self.jac_h(x_hat)
            hx = self.h(x_hat)
            v = y - hx - H @ (x_prior - x_hat)
            PHT = P_prior @ H.T
            S = H @ PHT + self.R
            K = PHT @ np.linalg.inv(S)
            x_hat = x_prior + K @ v
        
        x_hat_posterior = x_hat
        I_KH = self._I - K @ H
        P_posterior = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        P_posterior_inv = np.linalg.inv(P_posterior)
        
        return x_hat_posterior, P_posterior_inv

    def update_ukf_init(self, y, x_prior, P_prior):
        x = x_prior
        P = P_prior
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.h(s))
        sigmas_h = np.atleast_2d(sigmas_h)
        zp, S = UT(sigmas_h, self.points.Wm, self.points.Wc, self.R)
        SI = np.linalg.inv(S)
        Pxz = self.cross_variance(x, zp, self.sigmas_f, sigmas_h)
        K = np.dot(Pxz, SI)        # Kalman gain
        y = y - zp   # residual

        x = x + np.dot(K, y)
        P = P - np.dot(K, np.dot(S, K.T))
        return x, np.linalg.inv(P)
    
    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = sigmas_f[i] - x
            dz = sigmas_h[i] - z
            Pxz += self.points.Wc[i] * np.outer(dx, dz)
        return Pxz
    
    def predict(self, u=None):
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s, u)        
        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)

        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def update(self, y):
        lr = self.lr
        n_iterations = self.n_iterations
        
        # Initialize the first iteration step
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()

        if self.init_type == 'prior':
            x_hat, P_inv = x_hat_prior, P_inv_prior
        elif self.init_type == 'laplace':
            x_hat, P_inv = self.update_init(y, x_hat_prior, self.P.copy())
        elif self.init_type == 'iekf':
            x_hat, P_inv = self.update_iekf_init(y, x_hat_prior, self.P.copy(), self.iekf_max_iter)
        elif self.init_type == 'ukf':
            x_hat, P_inv = self.update_ukf_init(y, x_hat_prior, self.P.copy())
        else:
            raise ValueError
        
        is_positive_semidefinite(P_inv)

        for _ in range(n_iterations):          
            P = np.linalg.inv(P_inv)
            is_positive_semidefinite(P)

            E_hessian = cal_mean(lambda x : self.jac_h(x).T @ np.linalg.inv(self.R) @ self.jac_h(x), x_hat, P, self.points)
            P_inv_next = P_inv_prior + lr * E_hessian
            P_next = np.linalg.inv(P_inv_next)
            x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, y), x_hat, P, self.points) + P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
            if kld < self.threshold:
                P_inv = P_inv_next.copy()
                x_hat = x_hat_next.copy()
                break

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
