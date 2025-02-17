from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
from filterpy.kalman import unscented_transform as UT
import autograd.numpy as np
from autograd.numpy import dot
from copy import deepcopy

class UKF(UnscentedKalmanFilter):
    
    def __init__(self, model):
        super().__init__(
            fx = model.f,
            hx = model.h,
            dt = model.dt,
            dim_x = model.dim_x,
            dim_z = model.dim_y,
            points = MerweScaledSigmaPoints(model.dim_x, alpha=1e-3, beta=2.0, kappa=0)
        )

        self.Q = model.Q
        self.R = model.R

        self.x = model.x0
        self.P = model.P0
        
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
        # self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)
        
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
    
    def update(self, z):
        # z = self.get_y(wk, phik, dphik, yawk)
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R)
        self.SI = self.inv(self.S)
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)
        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()