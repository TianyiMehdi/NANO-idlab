import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from draw_utils import read_npz_files_in_s_folders, read_json_from_folders

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

path_to_results = './results/' + 'Robot'
data_files = read_npz_files_in_s_folders(path_to_results)
print(data_files.keys())

import matplotlib.pyplot as plt
import numpy as np

# 假设 data_files 已经定义，并且包含 'EKF' 和 'true_state', 'ekf_state'

true_state = data_files['EKF']['x_mc']
ekf_state = data_files['EKF']['x_hat_mc']

mc, time_length, num_states = ekf_state.shape

landmarks = np.array([[-1, 2], [-1, 10], [5, 1], [5, 10]])

for k in range(mc):
    plt.figure(figsize=(16, 12))
    
    # 绘制 true_state 和 ekf_state 的轨迹
    plt.plot(true_state[k, :, 0], true_state[k, :, 1], label='True State', color='blue', linewidth=2)
    plt.plot(ekf_state[k, :, 0], ekf_state[k, :, 1], label='EKF State', color='red', linestyle='--', linewidth=2)
    
    # 绘制 landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='black', marker='o', s=100, label='Landmarks', edgecolors='white')
    
    # 图例和标题
    plt.title(f'Trajectory for MC {k+1}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='best')
    
    # 保存图像
    plt.savefig(f'./figs/trajectory_mc{k+1}.png')
    
    