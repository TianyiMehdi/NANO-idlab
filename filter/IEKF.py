from filterpy.kalman import ExtendedKalmanFilter
import numpy as np



class IEKF(ExtendedKalmanFilter):
    def __init__(self, model, **filter_dict):
        super().__init__(dim_x=model.dim_x,
                         dim_z=model.dim_y,
                         )

        self.model_name = type(model).__name__
        self.f = model.f
        self.h = model.h
        self.jac_f = model.jac_f
        self.jac_h = model.jac_h
        self.max_iter = filter_dict['max_iter']

        self.Q = model.Q
        self.R = model.R

        self.x = model.x0
        self.P = model.P0 

    def predict(self, u=None):
        F = self.jac_f(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def update(self, y, max_iter=None):
        if max_iter == None:
            max_iter = self.max_iter

        x_hat = self.x
        P_prior = self.P_prior
        for i in range(max_iter):
            H = self.jac_h(self.x)
            hx = self.h(self.x)
            v = y - hx - H @ (x_hat - self.x)
            PHT = P_prior @ H.T
            self.S = H @ PHT + self.R
            self.K = PHT @ np.linalg.inv(self.S)
            # x_r= self.K @ (y - hx - H @ (x_hat - self.x))
            self.x = x_hat + self.K @ v
            # print(np.linalg.norm(self.x))
        I_KH = self._I - self.K @ H
        # self.P = P_prior - self.K @ self.S @ self.K.T
        self.P = (I_KH @ P_prior @ I_KH.T) + (self.K @ self.R @ self.K.T)
        
        # if self.model_name == 'Attitude':
        #     self.x = self.x / np.linalg.norm(self.x)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()