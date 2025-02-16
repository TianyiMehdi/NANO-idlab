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

# env = 'SinCos', 'sensor_net', 'Robot', 'Oscillator', 'Vehicle'
env = 'Toy_beta'

# Example usIn this subsection, we consider a non-autonomous system with control inputs, which is commonly used in robot localization tasksage
path_to_results = './results/' + env
data_files = read_npz_files_in_s_folders(path_to_results)
print(data_files.keys())

def get_error(name):
    error = data_files[name]['x_mc'] - data_files[name]['x_hat_mc']
    return error

def get_error_mean(name):
    error = np.mean(data_files[name]['x_mc'] - data_files[name]['x_hat_mc'], axis=0)
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
    '''
    1. Box Plot
    '''
    armse_ekf = np.mean(np.sqrt(np.mean((data_files['EKF']['x_mc'] -
                                         data_files['EKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_ukf = np.mean(np.sqrt(np.mean((data_files['UKF']['x_mc'] -
                                         data_files['UKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_IEKF = np.mean(np.sqrt(np.mean((data_files['IEKF']['x_mc'] -
                                          data_files['IEKF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_plf = np.mean(np.sqrt(np.mean((data_files['PLF']['x_mc'] -
                                         data_files['PLF']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_ggf = np.mean(np.sqrt(np.mean((data_files['NANO']['x_mc'] -
                                         data_files['NANO']['x_hat_mc']) ** 2, axis=(1))), axis=1)
    armse_values = [
        armse_ekf,
        armse_ukf,
        armse_IEKF,
        armse_plf,
        armse_ggf,
    ]

    # names = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    labels = ['EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    colors = ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    plt.figure(figsize=(12, 8))
    # plt.yscale('log')
    box = plt.boxplot(armse_values, vert=True, patch_artist=True,
                      showfliers=False)  # 'patch_artist=True' fills the box with color
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    for median in box['medians']:
        median.set_color('black')
    means = [np.mean(datapoint) for datapoint in armse_values]
    # 浅色的框，更重视均值，粗线
    plt.plot(range(1, len(armse_values) + 1), means, marker='o', 
             linestyle='--', color='red', label='Mean', linewidth=3, markersize=12)
    plt.xticks(ticks=range(1, len(armse_values) + 1), labels=labels, rotation=45, ha='right')
    plt.ylabel('Root Mean Square Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./figures/' + env + '/' + env + '.pdf', bbox_inches='tight')


    '''
    2. Error Curve Plot
    '''
    plt.rcParams['font.size'] = 48
    json_data = read_json_from_folders(path_to_results)

    if 'SinCos' in env:
        y_name = [r'$x_1$', r'$x_2$']
    elif 'Robot' in env:
        y_name = [r'$p_x$',  r'$p_y$']
    elif 'Lorenz' in env or 'Toy' in env:
        y_name = [r'$x_1$', r'$x_2$', r'$x_3$']
    elif 'SensorNet' in env:
        y_name = [r'$p_x$',  r'$p_y$', r'$\dot{p}_x$' , r'$\dot{p}_y$']
    elif 'Vehicle' in env:
        y_name = [r'$\delta$',  r'$\Omega$']
    elif 'Localization' in env:
        y_name = [r'$p_x$', r'$p_y$', r'$\phi$']
    else:
        y_name = [r'$x_1$', r'$x_2$']

    mc_index = 4
    # Now wiener_json_data contains the JSON content, keyed by folder name
    ekf_error = get_error('NANO')
    mc, time_length, num_states = ekf_error.shape
    # error_value = np.mean(ekf_error, axis=0)
    error_value = ekf_error[mc_index]
    max_error_value = np.max(error_value[20:], axis=0)
    min_error_value = np.min(error_value[20:], axis=0)
    
    pd_list = [[] for _ in range(num_states)]
    for tmp in range(len(labels)):  #
        for i in range(num_states):  # x_dim
            # for j in range(mc):  # mc
            pd_list[i].append(pd.DataFrame(
            # {'Error': get_error(labels[tmp])[j, :time_length, i],
                                            {'Error': get_error(labels[tmp])[mc_index, :time_length, i],
                                            'Algorithm': labels[tmp],
                                            'Step': np.arange(time_length),
                                            'Color': colors[tmp]
                                            }))
    
    for j in range(len(pd_list)):
        if not pd_list[j]:
            break
        pd_error = pd.concat(pd_list[j])
        pd_error = pd_error.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        palette = sns.color_palette(colors)
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error,
                          palette=palette, linewidth=4, dashes=False, legend=False)
        ax1.set_ylabel(y_name[j]+' Error')
        ax1.set_xlabel("Step")
        plt.xlim(1, time_length)
        if max_error_value[j] > 0:
            y_sup = 1.0 * max_error_value[j]
        else:
            y_sup = max_error_value[j] / 2

        if min_error_value[j] < 0:
            y_inf = 1.0 * min_error_value[j]
        else:
            y_inf = min_error_value / 2

        print(y_inf, y_sup)
        plt.ylim(y_inf, y_sup)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xticks()
        plt.yticks()
        plt.savefig('./figures/' + env + '/' + env + '_error_' + str(j+1) + '.pdf', bbox_inches='tight')

    
    '''
    3. RMS_Error Curve Plot
    '''
    # 对每个状态绘制一个 RMSE 图
    # 对每个状态绘制一个 RMSE 图
    for state_index in range(num_states):
        plt.figure(figsize=(16, 12))
        
        rmse_max = 0
        # 绘制每个算法的 RMSE 曲线
        for filter_idx, filter_name in enumerate(labels):
            # 计算该算法的 RMSE
            rmse = get_rms_error(filter_name, state_index)

            this_rmse_max = np.max(rmse[-100:])
            if this_rmse_max > rmse_max:
                rmse_max = this_rmse_max

            # 准备用于绘图的数据
            steps = np.arange(time_length)

            plt.plot(steps, rmse[:time_length], label=filter_name, color=colors[filter_idx], linewidth=4)

        plt.xlabel("Step")
        plt.ylabel(y_name[state_index] + ' RMSE')
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xlim(0, time_length)
        plt.ylim(0, rmse_max*1.3)

        plt.tight_layout()
        plt.savefig('./figures/' + env + '/' + env + '_rmse_' + str(state_index + 1) + '.pdf', bbox_inches='tight')
    

    '''
    4. State Curve Plot
    '''
    state_labels = ['True', 'EKF', 'UKF', 'IEKF', 'PLF', 'NANO']
    state_colors = ['C6'] + ['lightblue'] + ['C0'] + ['C1'] + ['C2'] + ['C3']
    
    pd_list = [[] for _ in range(num_states)]
    nano_state = get_state('NANO')
    true_state = np.mean(get_state('True'), axis=0)
    max_true_state = np.max(true_state[-100:], axis=0)
    min_true_state = np.min(true_state[-100:], axis=0)

    for tmp in range(len(state_labels)):
        for i in range(num_states):  # x_dim
            # for j in range(mc):  # mc
            pd_list[i].append(pd.DataFrame({'State': get_state(state_labels[tmp])[mc_index, :time_length, i],
                                            'Algorithm': state_labels[tmp],
                                            'Step': np.arange(time_length),
                                            'Color': state_colors[tmp]
                                            }))

    for j in range(len(pd_list)):
        if not pd_list[j]:
            break
        pd_state = pd.concat(pd_list[j])
        pd_state = pd_state.reset_index(drop=True)

        axes_pos = [0.2, 0.2, 0.6, 0.6]
        palette = sns.color_palette(state_colors)
        f1 = plt.figure(figsize=(16, 12), dpi=100)
        ax1 = f1.add_axes(axes_pos)
        g1 = sns.lineplot(x='Step', y="State", hue="Algorithm", style="Algorithm", data=pd_state,
                          palette=palette, linewidth=2, dashes=False, legend=False)
        ax1.set_ylabel(y_name[j])
        ax1.set_xlabel("Step")
        plt.xlim(1, time_length)
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.yticks()
        plt.xticks()
        plt.savefig('./figures/' + env + '/' + env + '_state_' + str(j+1) + '.pdf', bbox_inches='tight')
    
    
    '''
    5. Legend Plot
    '''
    linestyle = '-'
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, 
                        linestyle=linestyle, label=labels[i]) for i in range(len(labels))]
    f2 = plt.figure(figsize=(8, 4), dpi=100)
    ax2 = f2.add_axes([0, 0, 1, 1])
    ax2.legend(handles=legend_elements, loc='center', ncol=3)
    ax2.axis('off')
    plt.savefig('./figures/legend.pdf', bbox_inches='tight')
