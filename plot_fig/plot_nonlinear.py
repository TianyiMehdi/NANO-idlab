import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from draw_utils import read_npz_files_in_s_folders, read_json_from_folders, calculate_rmse, y_name_dict, get_error, get_state, get_rms_error
from matplotlib import font_manager
import os

simsun = font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

# 1. Set the parameters
language = 'English' # 'Chinese'
## env = 'SequenceForcasting', 'Localization', 'Oscillator', 'Attitude', 'GrowthModel'
env_ori = 'SequenceForcasting'
## noise_type = 'Gaussian', 'Laplace', 'Beta'
noise_type = 'Beta'
env = env_ori + '_' + noise_type
font_size_rmse = 24
font_size = 48

if 'Oscillator' in env:
    method_names = ['KF', 'UKF', 'NANO']
    colors = {'KF': 'lightblue', 'UKF': 'C0', 'NANO': 'C3'}

else:
    method_names = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    colors = {'EKF': 'lightblue', 'UKF': 'C0', 'IEKF': 'C1', 'PLF': 'C2', 'NANO': 'C3'}

mc_index = 4
time_length = 200


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.size'] = font_size_rmse 




if __name__ == '__main__':
    # 2. Load data
    path_to_results = './results/' + env
    data_files = read_npz_files_in_s_folders(path_to_results)
    print('Methods: ', data_files.keys())
    if set(method_names) != set(data_files.keys()):
        raise ValueError
    
    # 3. Plot figure

    '''
    1. Box Plot
    '''
    rmse_dict = dict()
    for method in method_names:
        x_mc = data_files[method]['x_mc']
        x_hat_mc = data_files[method]['x_hat_mc']
        rmse = calculate_rmse(x_mc, x_hat_mc) 
        
        rmse_dict[method] = rmse
    
    rmse_values = rmse_dict.values()
        
    plt.figure(figsize=(12, 8))
    box = plt.boxplot(rmse_values, vert=True, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    for median in box['medians']:
        median.set_color('black')
        
    armse = [np.mean(datapoint) for datapoint in rmse_values]
    plt.plot(range(1, len(armse) + 1), armse, marker='o', 
             linestyle='--', color='red', label='Mean', linewidth=3, markersize=12)
    plt.xticks(ticks=range(1, len(armse) + 1), labels=method_names, rotation=0, ha='right')
    
    if language == 'English':
        plt.ylabel('Root Mean Square Error')
    elif language == 'Chinese':
        plt.ylabel('均方根误差', fontproperties=simsun, fontsize= font_size_rmse)
        
    plt.grid(True)
    plt.tight_layout()
    save_path = './figures/' + env + '/'
    file_name = env + '.png'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + file_name, bbox_inches='tight')
        
        
    '''
    2. Error Curve Plot
    '''
    plt.rcParams['font.size'] = font_size
    
    json_data = read_json_from_folders(path_to_results)
    y_name = y_name_dict[env_ori]
    nano_error = get_error('NANO', data_files)
    mc, _, num_states = nano_error.shape
    

    error_value = nano_error[mc_index,:, :]
    max_error_value = np.max(error_value[20:], axis=0)
    min_error_value = np.min(error_value[20:], axis=0)
    
    pd_list = [[] for _ in range(num_states)]
    for method_name in method_names:  #
        for i in range(num_states):  # x_dim
            # for j in range(mc):  # mc
            pd_list[i].append(pd.DataFrame(
                            {'Error': get_error(method_name, data_files)[mc_index, :time_length, i],
                            'Algorithm': method_name,
                            'Step': np.arange(0, time_length),
                            'Color': colors[method_name]
                            }))
     
    for j in range(num_states):
        pd_error = pd.concat(pd_list[j])
        pd_error = pd_error.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        palette = sns.color_palette(colors.values())
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error,
                          palette=palette, linewidth=4, dashes=False, legend=False)
        
        if language == 'English':
            ax1.set_ylabel(y_name[j]+' Error')
            ax1.set_xlabel("Step")
        elif language == 'Chinese':
            ax1.set_ylabel(y_name[j]+' 误差', fontproperties=simsun, fontsize=font_size)
            ax1.set_xlabel("步数", fontproperties=simsun, fontsize=font_size)
        
        if max_error_value[j] > 0:
            y_sup = 1.25 * max_error_value[j]
        else:
            y_sup = max_error_value[j] / 2

        if min_error_value[j] < 0:
            y_inf = 1.25 * min_error_value[j]
        else:
            y_inf = min_error_value / 2

        print(y_inf, y_sup)
        plt.xlim(0, time_length)
        plt.ylim(y_inf, y_sup)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xticks()
        plt.yticks()
        plt.savefig('./figures/' + env + '/' + env + '_error_' + str(j+1) + '.png', bbox_inches='tight')

    
    '''
    3. RMS_Error Curve Plot
    '''
    # 对每个状态绘制一个 RMSE 图
    # 对每个状态绘制一个 RMSE 图
    for state_index in range(num_states):
        plt.figure(figsize=(16, 12))
        
        rmse_max = 0
        # 绘制每个算法的 RMSE 曲线
        for filter_idx, filter_name in enumerate(method_names):
            # 计算该算法的 RMSE
            rmse = get_rms_error(filter_name, state_index, data_files)

            this_rmse_max = np.max(rmse[-100:])
            if this_rmse_max > rmse_max:
                rmse_max = this_rmse_max

            # 准备用于绘图的数据
            steps = np.arange(0, time_length)

            plt.plot(steps, rmse[:time_length], label=filter_name, color=colors[filter_name], linewidth=4)

        plt.xlabel("Step")
        plt.ylabel(y_name[state_index] + ' RMSE')
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xlim(0, time_length)
        plt.ylim(0, rmse_max*1.3)

        plt.tight_layout()
        plt.savefig('./figures/' + env + '/' + env + '_rmse_' + str(state_index + 1) + '.png', bbox_inches='tight')
    

    '''
    4. State Curve Plot
    '''
    state_labels = ['True'] + method_names
    state_colors = colors|{'True': 'Black'}
    
    pd_list = [[] for _ in range(num_states)]
    
    true_state = get_state('True', data_files)[mc_index]
    max_true_state = np.max(true_state[-100:], axis=0)
    min_true_state = np.min(true_state[-100:], axis=0)

    for label in state_labels:
        for i in range(num_states):  # x_dim
            # for j in range(mc):  # mc
            pd_list[i].append(pd.DataFrame({'State': get_state(label, data_files)[mc_index, :time_length, i],
                                            'Algorithm': label,
                                            'Step': np.arange(0, time_length),
                                            'Color': state_colors[label]
                                            }))

    for j in range(len(pd_list)):
        pd_state = pd.concat(pd_list[j])
        pd_state = pd_state.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        palette = sns.color_palette(state_colors.values())
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="State", hue="Algorithm", style="Algorithm", data=pd_state,
                          palette=palette, linewidth=2, dashes=False, legend=False)
        ax1.set_ylabel(y_name[j])
        if language == 'English':
            ax1.set_xlabel("Step")
        elif language == 'Chinese':
            ax1.set_xlabel("步数", fontproperties=simsun, fontsize=font_size)
            
        plt.xlim(0, time_length)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.yticks()
        plt.xticks()
        plt.savefig('./figures/' + env + '/' + env + '_state_' + str(j+1) + '.png', bbox_inches='tight')
    
    
    '''
    5. Legend Plot
    '''
    # plt.rcParams['font.size'] = font_size_rmse
    linestyle = '-'
    legend_elements = [Line2D([0], [0], color=colors[name], lw=4, 
                        linestyle=linestyle, label=name) for name in method_names]
    f2 = plt.figure(figsize=(8, 4), dpi=100)
    ax2 = f2.add_axes([0, 0, 1, 1])
    ax2.axis('off')
    if 'Oscillator' in env:
        ax2.legend(handles=legend_elements, loc='center', ncol=3)
        plt.savefig('./figures/legend_linear.png', bbox_inches='tight')
    else:
        ax2.legend(handles=legend_elements, loc='center', ncol=5)
        plt.savefig('./figures/legend.png', bbox_inches='tight')


    legend_elements = [Line2D([0], [0], color=state_colors[name], lw=4, 
                        linestyle=linestyle, label=name) for name in state_labels]
    f2 = plt.figure(figsize=(8, 4), dpi=100)
    ax2 = f2.add_axes([0, 0, 1, 1])
    ax2.axis('off')
    if 'Oscillator' in env:
        ax2.legend(handles=legend_elements, loc='center', ncol=4)
        plt.savefig('./figures/legend_state_linear.png', bbox_inches='tight')
    else:
        ax2.legend(handles=legend_elements, loc='center', ncol=3)
        plt.savefig('./figures/legend_state.png', bbox_inches='tight')
