from einops import rearrange, reduce
from typing import Dict
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
from filterpy.kalman import unscented_transform as UT
import autograd.numpy as np
from autograd.numpy import eye, ones, zeros, dot, isscalar, outer
from copy import deepcopy
from .utils import is_positive_semidefinite, kl_divergence

class PLF(UnscentedKalmanFilter):
    
    def __init__(self, model, **filter_dict):
        super().__init__(
            fx = model.f,
            hx = model.h,
            dt = model.dt,
            dim_x = model.dim_x,
            dim_z = model.dim_y,
            # points = JulierSigmaPoints(model.dim_x, kappa=0)
            points = MerweScaledSigmaPoints(model.dim_x, alpha=0.3, beta=2.0, kappa=0)
        )

        self.model_name = type(model).__name__
        self.Q = model.Q
        self.R = model.R

        self.x = model.x0
        self.P = model.P0

        self.max_iter = filter_dict['max_iter']
        
    def compute_process_sigmas(self, u=None, dt=None, fx=None):

        if fx is None:
            fx = self.fx
        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, u)
        
    def predict(self, u=None):
        self.compute_process_sigmas(u)
        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                        self.x_mean, self.residual_x)
        
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
    
    def update(self, z):
        x_hat = self.x_prior
        P_hat = self.P_prior
        x_hat_old = 100 * np.ones_like(x_hat)
        P_hat_old = 100 * np.eye(self._dim_x)
        iter = 0
        sigmas = self.sigmas_f
        # max_iter:5
        while kl_divergence(x_hat_old, P_hat_old, x_hat, P_hat) >= 1e-2 and iter <= self.max_iter:
            iter += 1
            x_hat_old = x_hat
            P_hat_old = P_hat
            sigmas_h = zeros((self._num_sigmas, self._dim_z))
            for i, s in enumerate(sigmas):
                sigmas_h[i] = self.hx(s)
            zp, Pz = UT(sigmas_h, self.Wm, self.Wc, noise_cov=None)
            Pxz = zeros((self._dim_x, self._dim_z))
            for i in range(sigmas.shape[0]):
                Pxz += self.Wc[i] * np.outer(sigmas[i] - x_hat, sigmas_h[i] - zp)
            H = Pxz.T @ np.linalg.inv(P_hat)
            b = zp - H @ x_hat
            Omega = Pz - H @ P_hat @ H.T

            x_hat = self.x_prior + self.P_prior @ H.T @ np.linalg.inv(H @ self.P_prior @ H.T + Omega + self.R) @ (z - H @ self.x_prior - b)
            P_hat = self.P_prior - self.P_prior @ H.T @ np.linalg.inv(H @ self.P_prior @ H.T + Omega + self.R) @ H @ self.P_prior
            # x_hat = x_hat / np.linalg.norm(x_hat)
            sigmas = self.points_fn.sigma_points(x_hat, P_hat)
        
        # if self.model_name == 'Attitude':
        #     x_hat = x_hat / np.linalg.norm(x_hat)
        self.x = x_hat
        self.P = P_hat
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()