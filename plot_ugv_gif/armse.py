import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
# import Patch


import sys
sys.path.append('../../')


plt.style.use('/home/zhangtianyi/Gibss-Gaussian-Filtering/style.mpl')

def read_npz_files_in_s_folders(directory):
    data_files = {}
    for item in os.scandir(directory):
        if item.is_dir() and item.name.endswith(''):
            npz_path = os.path.join(item.path, 'data.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                data_files[item.name] = data
    
    return data_files

data_files = read_npz_files_in_s_folders('/home/zhangtianyi/Gibss-Gaussian-Filtering/plot_ugv_gif/data')
print(data_files.keys())

armse_ekf = np.mean(np.sqrt(np.mean((data_files['EKF']['x_mc'] - 
                                             data_files['EKF']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_ukf = np.mean(np.sqrt(np.mean((data_files['UKF']['x_mc'] - 
                                             data_files['UKF']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_iekf = np.mean(np.sqrt(np.mean((data_files['iEKF']['x_mc'] - 
                                             data_files['iEKF']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_plf = np.mean(np.sqrt(np.mean((data_files['PLF']['x_mc'] - 
                                             data_files['PLF']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_ggf = np.mean(np.sqrt(np.mean((data_files['GGF']['x_mc'] - 
                                             data_files['GGF']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_beta_5e_3 = np.mean(np.sqrt(np.mean((data_files['GGF_beta5e-3']['x_mc'] - 
                                             data_files['GGF_beta5e-3']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_beta_1e_2 = np.mean(np.sqrt(np.mean((data_files['GGF_beta1e-2']['x_mc'] - 
                                             data_files['GGF_beta1e-2']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_beta_2e_2 = np.mean(np.sqrt(np.mean((data_files['GGF_beta2e-2']['x_mc'] - 
                                             data_files['GGF_beta2e-2']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_huber_1 = np.mean(np.sqrt(np.mean((data_files['GGF_huber1']['x_mc'] - 
                                             data_files['GGF_huber1']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_huber_2 = np.mean(np.sqrt(np.mean((data_files['GGF_huber2']['x_mc'] - 
                                             data_files['GGF_huber2']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_huber_3 = np.mean(np.sqrt(np.mean((data_files['GGF_huber3']['x_mc'] - 
                                             data_files['GGF_huber3']['x_hat_mc'])**2, axis=(1))), axis=1)  
armse_weight_5 = np.mean(np.sqrt(np.mean((data_files['GGF_weight5']['x_mc'] - 
                                             data_files['GGF_weight5']['x_hat_mc'])**2, axis=(1))), axis=1)
armse_weight_8 = np.mean(np.sqrt(np.mean((data_files['GGF_weight8']['x_mc'] - 
                                             data_files['GGF_weight8']['x_hat_mc'])**2, axis=(1))), axis=1) 
armse_weight_10 = np.mean(np.sqrt(np.mean((data_files['GGF_weight10']['x_mc'] - 
                                             data_files['GGF_weight10']['x_hat_mc'])**2, axis=(1))), axis=1)

                                  
armse_values = [
    armse_ekf.mean(),
    armse_ukf.mean(),
    armse_iekf.mean(),
    armse_plf.mean(),
    armse_ggf.mean(),
    armse_beta_5e_3.mean(),
    armse_beta_1e_2.mean(),
    armse_beta_2e_2.mean(),
    armse_huber_1.mean(),
    armse_huber_2.mean(),
    armse_huber_3.mean(),
    armse_weight_5.mean(),
    armse_weight_8.mean(),
    armse_weight_10.mean(),
]
# print(armse_values.keys())
# print(armse_values.items())

print(armse_values)