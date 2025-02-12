class Model:

    def __init__(self, noise_type='Gaussian'):
        self.noise_type = noise_type
    
    def f(self, x, u=None):
        pass

    def h(self, x):
        pass

    def jac_f(self, x, u=None):
        pass

    def jac_h(self, x):
        pass

    def f_withnoise(self, x, u=None):
        pass

    def h_withnoise(self, x):
        pass
    
    