import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

from matplotlib.patches import Patch
from matplotlib.pyplot import Line2D

# sys.path.append('../../')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.size'] = 24


def read_npz_files_in_s_folders(directory):
    data_files = {}
    for item in os.scandir(directory):
        if item.is_dir() and item.name.endswith(''):
            npz_path = os.path.join(item.path, 'data.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                data_files[item.name] = data

    return data_files


def read_json_from_folders(base_path):
    """
    Reads JSON files from folders within the base_path. It also reads a JSON file
    located directly in the base_path. The data is stored in a dictionary keyed
    by the folder name.

    :param base_path: The base directory that contains the folders and JSON file.
    :return: A dictionary with the folder name as the key and the JSON content as the value.
    """
    json_data = {}

    # Check for a JSON file in the base directory
    for item in os.listdir(base_path):
        if item.endswith('.json'):
            file_path = os.path.join(base_path, item)
            with open(file_path, 'r') as json_file:
                # Assuming the file name format is "FolderName-Description.json"
                folder_name = item.split('-')[0]
                json_data[folder_name] = json.load(json_file)

    # Read JSON data from each folder
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            # Look for a JSON file inside this folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as json_file:
                        json_data[folder_name] = json.load(json_file)

    return json_data

start_step = 0


def get_error(name):
    error = data_files[name]['x_mc'] - data_files[name]['x_hat_mc']
    return error

def get_rms_error(name, state_index):
    x_mc = data_files[name]['x_mc']
    x_hat_mc = data_files[name]['x_hat_mc']
    rms_error = np.sqrt(np.mean((x_mc[:, :, state_index] - x_hat_mc[:, :, state_index])**2, axis=0))
    return rms_error

def get_state(name):
    if name == 'True':
        return data_files['NANO']['x_mc']
    else:
        return data_files[name]['x_hat_mc']

if __name__ == '__main__':
    # env = 'sin_cos', 'sensor_net', 'robot', 'oscillator', 'vehicle'
    env = 'robot'

    # Example usIn this subsection, we consider a non-autonomous system with control inputs, which is commonly used in robot localization tasksage
    data_files = read_npz_files_in_s_folders('../results/' + env)
    print(data_files.keys())

    armse_ekf = np.mean(np.sqrt(np.mean((data_files['EKF']['x_mc'] -
                                         data_files['EKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_ukf = np.mean(np.sqrt(np.mean((data_files['UKF']['x_mc'] -
                                         data_files['UKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_iekf = np.mean(np.sqrt(np.mean((data_files['iEKF']['x_mc'] -
                                          data_files['iEKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_plf = np.mean(np.sqrt(np.mean((data_files['PLF']['x_mc'] -
                                         data_files['PLF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_ggf = np.mean(np.sqrt(np.mean((data_files['NANO']['x_mc'] -
                                         data_files['NANO']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_values = [
        armse_ekf,
        armse_ukf,
        armse_iekf,
        armse_plf,
        armse_ggf,
    ]

    xlab1 = r'$\beta$'
    variable_names = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    label = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    plt.figure(figsize=(12, 8))
    # plt.yscale('log')
    box = plt.boxplot(armse_values, vert=True, patch_artist=True,
                      showfliers=False)  # 'patch_artist=True' fills the box with color
    colors = ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
    means = [np.mean(datapoint) for datapoint in armse_values]
    plt.plot(range(1, len(armse_values) + 1), means, marker='s', linestyle='--', color='black', label='Mean')
    plt.xticks(ticks=range(1, len(armse_values) + 1), labels=variable_names, rotation=45, ha='right')
    # plt.xlabel(xlab1)
    plt.ylabel('Root Mean Square Error')
    # plt.ylim(ylims[i])
    plt.grid(True)
    legend_elements = [Patch(facecolor=colors[0], label=label[0]),
                       Patch(facecolor=colors[1], label=label[1]),
                       Patch(facecolor=colors[2], label=label[2]),
                       Patch(facecolor=colors[3], label=label[3]),
                       Patch(facecolor=colors[4], label=label[4]), ]
    # plt.legend(handles=legend_elements, loc='upper right')
    # plt.show()
    plt.tight_layout()
    plt.savefig('../figures_02_11/' + env + '/' + env + '.pdf', bbox_inches='tight')


    plt.rcParams['font.size'] = 48
    # Usage
    # Replace 'path_to_wiener_results' with the actual path to the 'wiener_results' directory
    path_to_results = '../results/' + env
    json_data = read_json_from_folders(path_to_results)

    # Now wiener_json_data contains the JSON content, keyed by folder name

    '''
    2. Error Curve Plot
    '''
    name = ['EKF', 'UKF', 'iEKF', 'PLF', 'NANO']
    # name = ['ekf_s', 'ukf_s', 'huberukf_s', 'iekf_s', 'convekf0.5_s', 'convukf1_s']
    label = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    # colors = ['lightblue']+['C0']+['C1']+['C2'] + ['C4']+['C6']
    colors = ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    pd0_list, pd1_list, pd2_list, pd3_list, pd4_list, pd5_list, pd6_list = [], [], [], [], [], [], []
    pd_list = [pd0_list, pd1_list, pd2_list, pd3_list, pd4_list, pd5_list, pd6_list]
    ekf_error = get_error('NANO')
    time_length = ekf_error.shape[1]
    # print(ekf_error.shape)
    for tmp in range(len(name)):  #
        # if tmp == 2:
        #     continue
        for i in range(ekf_error.shape[2]):  # x_dim
            for j in range(ekf_error.shape[0]):  # mc
                pd_list[i].append(pd.DataFrame({'Error': get_error(name[tmp])[j, :, i],
                                                'Algorithm': label[tmp],
                                                'Step': np.arange(time_length),
                                                'Color': colors[tmp]
                                                }))
    # y_name = []
    # env = 'sin_cos', 'sensor_net', 'robot', 'oscillator'
    if 'sensor_net' in env:
        y_name = [r'$p_x$', r'$\dot{p}_x$',  r'$p_y$', r'$\dot{p}_y$']
    elif 'robot' in env:
        y_name = [r'$p_x$',  r'$p_y$']
    elif 'vehicle' in env:
        y_name = [r'$\delta$',  r'$\Omega$']
    else:
        y_name = [r'$x_1$', r'$x_2$']
    
    # font = 30
    for j in range(len(pd_list)):
        if not pd_list[j]:
            break
        pd_error = pd.concat(pd_list[j])
        pd_error = pd_error.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        BETA = [0.0001]
        palette = sns.color_palette(colors)
        linestyle = '-'
        legend_elements = [Line2D([0], [0], color=colors[i], lw=4, linestyle=linestyle, label=label[i]) for i in
                           range(len(label))]
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error,
                          palette=palette, linewidth=4, dashes=False, legend=False)
        # ax1.set_ylabel('Error', fontsize=font)
        ax1.set_ylabel(y_name[j]+' Error')
        ax1.set_xlabel("Step")
        plt.xlim(0, time_length)
        # plt.ylim(-1, 3)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        # ax1.set_xticks(4 * np.arange(6))
        # ax1.set_xticklabels(('0', '4', '8', '12', '16', '20'), fontsize=font)
        handles, labels = ax1.get_legend_handles_labels()
        # ax1.legend(handles=handles, labels=labels, fontsize=font)
        # ax1.legend(legend_elements)
        plt.yticks()
        plt.xticks()
        plt.savefig('../figures_02_11/' + env + '/' + env + '_error_' + str(j+1) + '.pdf', bbox_inches='tight')

    
    '''
    3. RMS_Error Curve Plot
    '''
    name = ['EKF', 'UKF', 'iEKF', 'PLF', 'NANO']
    colors = ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    num_states = ekf_error.shape[2]
    # 对每个状态绘制一个 RMSE 图
    # 对每个状态绘制一个 RMSE 图
    for state_index in range(num_states):
        plt.figure(figsize=(16, 12))

        # 绘制每个算法的 RMSE 曲线
        for filter_idx, filter_name in enumerate(name):
            # 计算该算法的 RMSE
            rmse = get_rms_error(filter_name, state_index)

            # 准备用于绘图的数据
            steps = np.arange(time_length)

            # 绘制 RMSE 曲线，按算法区分颜色
            plt.plot(steps, rmse, label=filter_name, color=colors[filter_idx], linewidth=4)

        # 设置图像标签和标题
        plt.xlabel("Step")
        plt.ylabel('RMSE-' + y_name[state_index])  # 使用 TeX 公式表示RMSE
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xlim(0, time_length)

        # 添加图例
        # plt.legend(title='Algorithms')

        # 保存图像
        plt.tight_layout()
        plt.savefig('../figures_02_11/' + env + '/' + env + '_rmse_' + str(state_index + 1) + '.pdf', bbox_inches='tight')
    

    '''
    4. State Curve Plot
    '''
    name = ['True', 'EKF', 'UKF', 'iEKF', 'PLF', 'NANO']
    # name = ['ekf_s', 'ukf_s', 'huberukf_s', 'iekf_s', 'convekf0.5_s', 'convukf1_s']
    label = ['True', 'EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    # colors = ['lightblue']+['C0']+['C1']+['C2'] + ['C4']+['C6']
    colors = ['C6'] + ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    pd0_list, pd1_list, pd2_list, pd3_list, pd4_list, pd5_list, pd6_list = [], [], [], [], [], [], []
    pd_list = [pd0_list, pd1_list, pd2_list, pd3_list, pd4_list, pd5_list, pd6_list]
    nano_state = get_state('NANO')
    time_length = nano_state.shape[1]
    # print(ekf_error.shape)
    for tmp in range(len(name)):  #
        # if tmp == 2:
        #     continue
        for i in range(nano_state.shape[2]):  # x_dim
            for j in range(nano_state.shape[0]):  # mc
                pd_list[i].append(pd.DataFrame({'State': get_state(name[tmp])[j, :, i],
                                                'Algorithm': label[tmp],
                                                'Step': np.arange(time_length),
                                                'Color': colors[tmp]
                                                }))
    # y_name = []
    # env = 'sin_cos', 'sensor_net', 'robot', 'oscillator'
    if 'sensor_net' in env:
        y_name = [r'$p_x$', r'$\dot{p}_x$',  r'$p_y$', r'$\dot{p}_y$']
    elif 'robot' in env:
        y_name = [r'$p_x$',  r'$p_y$']
    elif 'vehicle' in env:
        y_name = [r'$\delta$',  r'$\Omega$']
    else:
        y_name = [r'$x_1$', r'$x_2$']
    
    # font = 30
    for j in range(len(pd_list)):
        if not pd_list[j]:
            break
        pd_state = pd.concat(pd_list[j])
        pd_state = pd_state.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        BETA = [0.0001]
        palette = sns.color_palette(colors)
        linestyle = '-'
        legend_elements = [Line2D([0], [0], color=colors[i], lw=4, linestyle=linestyle, label=label[i]) for i in
                           range(len(label))]
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="State", hue="Algorithm", style="Algorithm", data=pd_state,
                          palette=palette, linewidth=2, dashes=False, legend=False)
        # ax1.set_ylabel('Error', fontsize=font)
        ax1.set_ylabel(y_name[j])
        ax1.set_xlabel("Step")
        plt.xlim(0, time_length)
        # plt.ylim(-1, 3)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        # ax1.set_xticks(4 * np.arange(6))
        # ax1.set_xticklabels(('0', '4', '8', '12', '16', '20'), fontsize=font)
        handles, labels = ax1.get_legend_handles_labels()
        # ax1.legend(handles=handles, labels=labels, fontsize=font)
        # ax1.legend(legend_elements)
        plt.yticks()
        plt.xticks()
        plt.savefig('../figures_02_11/' + env + '/' + env + '_state_' + str(j+1) + '.pdf', bbox_inches='tight')
    
    f2 = plt.figure(figsize=(8, 4), dpi=100)
    ax2 = f2.add_axes([0, 0, 1, 1])
    ax2.legend(handles=legend_elements, loc='center', ncol=3)
    ax2.axis('off')
    plt.savefig('../figures_02_11/legend_state.pdf', bbox_inches='tight')
