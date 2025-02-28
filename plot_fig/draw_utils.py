import os
import json
import numpy as np

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


y_name_dict = {'Oscillator': [r'$x_1$', r'$x_2$'],
               'Attitude': ['Roll', 'Pitch', 'Yaw'],
               'Localization': [r'$p_x$', r'$p_y$', r'$\phi$'],
               'SequenceForcasting': [r'$x_1$', r'$x_2$'],
               'GrowthModel': [r'$x_1$', r'$x_2$', r'$x_3$']}

def get_error(name, data_files):
    error = data_files[name]['x_mc'] - data_files[name]['x_hat_mc']
    return error

def get_state(name, data_files):
    if name == 'True':
        return data_files['NANO']['x_mc']
    else:
        return data_files[name]['x_hat_mc']
    
def get_rms_error(name, state_index, data_files):
    x_mc = data_files[name]['x_mc']
    x_hat_mc = data_files[name]['x_hat_mc']
    rms_error = np.sqrt(np.mean((x_mc[:, :, state_index] - x_hat_mc[:, :, state_index])**2, axis=0))
    return rms_error

def calculate_rmse(x_mc, x_hat_mc):
    rmse = np.sqrt(np.mean((x_mc - x_hat_mc)**2, axis=(1, 2))) 
    return rmse