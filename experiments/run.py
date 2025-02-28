from tqdm import tqdm
import time
import autograd.numpy as np
import importlib

def run_filter(N_exp, steps, model_name, noise_type, filter_name, 
               x_hat0, filter_dict, control_input=None):
    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []
    for _ in tqdm(range(N_exp)):
        x_list, y_list, x_hat_list, run_time = [], [], [], []
        
        # model = Oscillator(noise_name)
        module = importlib.import_module(f"environ.{camel_to_snake(model_name)}")
        Model_class = getattr(module, model_name)
        model = Model_class(noise_type)

        # initialize system
        x = model.x0
        y = model.h_withnoise(x)

        # filter = EKF(model)
        module = importlib.import_module(f"filter.{filter_name}")
        Filter_class = getattr(module, filter_name)
        filter = Filter_class(model, **filter_dict)
        
        filter.x = x_hat0

        x_list.append(x)
        y_list.append(y)
        x_hat_list.append(filter.x)

        for i in range(1, steps):
            # generate data
            if control_input is None:
                u = None
            else: 
                u = control_input[i]
            x = model.f_withnoise(x, u)
            # x = model.f(x, u)
            y = model.h_withnoise(x)
            x_list.append(x)
            y_list.append(y)

            time1 = time.time()
            
            # perform filtering
            filter.predict(u)
            filter.update(y)
            time2 = time.time()
            x_hat_list.append(filter.x)
            
            run_time.append(time2 - time1)

        x_mc.append(np.array(x_list))
        y_mc.append(np.array(y_list))
        x_hat_mc.append(np.array(x_hat_list))
        all_time.append(np.mean(run_time))

    x_mc = np.array(x_mc)
    y_mc = np.array(y_mc)
    x_hat_mc = np.array(x_hat_mc)
    mean_time = np.mean(all_time)

    data_dict = {'x_mc': x_mc, 'y_mc': y_mc, 'x_hat_mc': x_hat_mc, 'mean_time': mean_time}
    
    if model_name == 'Attitude':
        
        data_dict['x_mc'] = data_dict['x_mc'] * 180.0 / np.pi
        data_dict['x_hat_mc'] = data_dict['x_hat_mc'] * 180.0 / np.pi
    
    return data_dict


def camel_to_snake(name):
    name_chars = list(name)

    new_chars = []

    for i, char in enumerate(name_chars):
        # 检查字符是否大写
        if char.isupper():
            # 如果不是首字符，并且前一个字符不是大写或下一个字符是小写，则在前面添加下划线
            if i != 0 and (not name_chars[i - 1].isupper() or (i + 1 < len(name_chars) and name_chars[i + 1].islower())):
                new_chars.append('_')
            new_chars.append(char.lower())
        else:
            new_chars.append(char)

    return ''.join(new_chars)