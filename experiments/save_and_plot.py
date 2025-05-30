from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import autograd.numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
directory_path = os.path.dirname(directory_path)
# plt.rcParams

def calculate_rmse(x_mc, x_hat_mc):
    """
    Calculate the Average Root Mean Square Error (RMSE) for each state.

    :param x_mc: Numpy array with shape (num_experiments, trajectory_length, num_states) representing the true states.
    :param x_hat_mc: Numpy array with shape (num_experiments, trajectory_length, num_states) representing the estimated states.
    :return: A numpy array containing the average RMSE for each state.
    """
    squared_errors = (x_mc - x_hat_mc) ** 2

    threshold=100
    # if np.sqrt(np.mean(squared_errors[-1], axis=0)).all()>threshold:
    #     raise ValueError
    rmse_sum=0
    index=0

    for squared_error in squared_errors:
        rmse=np.sqrt(np.mean(squared_error, axis=0))
        # if (rmse<threshold).all():#剔除实验异常值
        rmse_sum+=rmse
        index+=1
    rmse_perstate=rmse_sum/index
    return rmse_perstate #先求总的平均值再开根号 ARMSE 每一次实验都开个根号再求平均值


def save_per_exp(data_dict, args_dict, filter_dict):
    data_dir = args_dict['result_dir']
    x_mc = data_dict['x_mc']
    x_hat_mc = data_dict['x_hat_mc']
    x_rmse = calculate_rmse(x_mc, x_hat_mc)
    # if args_dict['model_name'] == 'Attitude':
    #     x_rmse = (180 / np.pi) * x_rmse
    print('RMSE:', x_rmse)
    print('time:', data_dict['mean_time'])
    data_dict['x_rmse'] = x_rmse
    args_dict['x_rmse'] = x_rmse.tolist()
    args_dict['time'] = data_dict['mean_time']
    args_dict.update(filter_dict) 
    
    # Get the current date and time in the format Year-Month-Day-Hour-Minute
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args_dict['current_time'] = current_time
    
    model_name = args_dict['model_name'] + '_' + args_dict['noise_type']
    filter_name = args_dict['filter_name']
    # Create a directory named after the current date and time
    if data_dir is None:
        data_dir = f"{directory_path}/results/{model_name}/{filter_name}"
    
    os.makedirs(data_dir, exist_ok=True)

    # Create a file name for parameters based on 'filter_name', 'model_name', and 'noise_name' from the parameters dict
    parameter_file_name = f"{args_dict.get('filter_name', 'unknown')}-{args_dict.get('model_name', 'unknown')}-{args_dict.get('noise_name', 'unknown')}.json"
    data_file_name = "data.npz"

    # Save parameters to a JSON file in the created directory
    with open(os.path.join(data_dir, parameter_file_name), 'w') as file:
        json.dump(args_dict, file, indent=4)

    # Convert the data dictionary to arrays and save to an NPZ file in the created directory
    np.savez(os.path.join(data_dir, data_file_name), **data_dict)

    plot_state_rmse_error(data_dir, args_dict['filter_name'], **data_dict)
    plot_state(data_dir, args_dict['filter_name'], **data_dict)

    return f"Files saved in directory: {data_dir}"


def plot_state(data_dir, filter_name, **data_dict):
    x_mc = data_dict['x_mc']
    x_hat_mc = data_dict['x_hat_mc']
    num_experiments, num_steps, num_states = x_mc.shape

    # 使用 Seaborn 的样式
    plt.style.use(f"{directory_path}/style.mpl")

    for k in range(10):
    # 对每个状态绘制 x_mc 和 x_hat_mc 的均值曲线
        for state_index in range(num_states):
            x_mc_sample = x_mc[k, :, state_index]
            x_hat_mc_sample = x_hat_mc[k, :, state_index]

            # 准备用于 Seaborn 绘图的数据
            data = pd.DataFrame({
                'Step': np.arange(num_steps),
                'x_mc': x_mc_sample,
                'x_hat_mc': x_hat_mc_sample,
            })

            # 绘制均值曲线（带标准差）
            plt.figure(figsize=(8, 6))
            sns.lineplot(x='Step', y='x_mc', data=data, label='True', ci=None)
            sns.lineplot(x='Step', y='x_hat_mc', data=data, label='Estimate', ci=None)

            plt.xlabel("Step")
            plt.ylabel(r'$x_{}$ Value'.format(state_index + 1))  # 使用 TeX 公式
            plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
            plt.xlim(0, num_steps - 1)

            # 添加图例
            plt.legend()

            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, f'{filter_name}_mc_{k}_state_{state_index + 1}_comparison.png'))


def plot_state_rmse_error(data_dir, filter_name, **data_dict):
    x_mc = data_dict['x_mc']
    x_hat_mc = data_dict['x_hat_mc']
    num_experiments, num_steps, num_states = x_mc.shape

    # 使用 Seaborn 的样式
    plt.style.use(f"{directory_path}/style.mpl")

    # 对每个状态绘制一个 RMSE 图
    for state_index in range(num_states):
        # 计算 RMSE
        rmse = np.sqrt(np.mean((x_mc[:, :, state_index] - x_hat_mc[:, :, state_index])**2, axis=0))

        # 准备用于 Seaborn 绘图的数据
        data = pd.DataFrame({
            'Step': np.arange(num_steps),
            'RMSE': rmse
        })

        # 绘制 RMSE 曲线
        plt.figure(figsize=(8, 6))
        sns.lineplot(x='Step', y='RMSE', data=data)

        plt.xlabel("Step")
        plt.ylabel(f'$RMSE-x_{state_index + 1}$')  # 使用 TeX 公式表示RMSE
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.xlim(0, num_steps - 1)

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'{filter_name}_state_{state_index + 1}_rmse.png'))
        plt.close()

def plot_box_methods(data_dicts):
    # box plot
    pass