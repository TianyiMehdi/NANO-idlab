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
    return data_dict


def run_filter_attitude(N_exp, steps, model_name, noise_type, filter_name, 
               x_hat0, filter_dict, control_input=None):
    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []
    gyro_std = 5/180*np.pi
    u = np.zeros(3)
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
        
        if x_hat0 is None:
            filter.x = x
        else:
            filter.x = x_hat0

        x_list.append(180 / np.pi * x)
        y_list.append(y)
        x_hat_list.append(180 / np.pi * filter.x)

        for i in range(1, steps):
            # generate data
            if control_input is None:
                u = None
            else: 
                u = control_input[i]
                # u = u + gyro_std*np.random.randn(3)
            x = model.f_withnoise(x, u)
            # x = model.f(x, u)
            y = model.h_withnoise(x)
            x_list.append(180 / np.pi * x)
            # print(np.linalg.norm(x))
            y_list.append(y)

            time1 = time.time()
            
            # perform filtering
            filter.predict(u)
            filter.update(y)
            time2 = time.time()
            x_hat_list.append(180 / np.pi * filter.x)
            
            run_time.append(time2 - time1)

        x_array = np.array(x_list)
        y_array = np.array(y_list)
        x_hat_array = np.array(x_hat_list)

        rmse = np.mean(np.sqrt(np.mean((x_array -
                                         x_hat_array) ** 2, axis=(0))), axis=0)
        # print(rmse)
        # if rmse < 8:
        x_mc.append(x_array)
        y_mc.append(y_array)
        x_hat_mc.append(x_hat_array)
        all_time.append(np.mean(run_time))

    x_mc = np.array(x_mc)
    y_mc = np.array(y_mc)
    x_hat_mc = np.array(x_hat_mc)
    mean_time = np.mean(all_time)
    print(x_mc.shape)

    data_dict = {'x_mc': x_mc, 'y_mc': y_mc, 'x_hat_mc': x_hat_mc, 'mean_time': mean_time}
    return data_dict


def run_filter_attitude_nl(N_exp, steps, model_name, noise_type, filter_name, 
               x_hat0, filter_dict, control_input=None):
    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []
    gyro_std = 5/180*np.pi
    u = np.zeros(3)
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
        
        if x_hat0 is None:
            filter.x = x
        else:
            filter.x = x_hat0

        x_list.append(180 / np.pi * quat_to_euler(x))
        y_list.append(y)
        x_hat_list.append(180 / np.pi * quat_to_euler(filter.x))

        for i in range(1, steps):
            # generate data
            if control_input is None:
                u = None
            else: 
                # u = control_input[i]
                u = u + gyro_std*np.random.randn(3)
            x = model.f_withnoise(x, u)
            # x = model.f(x, u)
            y = model.h_withnoise(x)
            x_list.append(180 / np.pi * quat_to_euler(x))
            # print(np.linalg.norm(x))
            y_list.append(y)

            time1 = time.time()
            
            # perform filtering
            filter.predict(u)
            filter.update(y)
            time2 = time.time()
            x_hat_list.append(180 / np.pi * quat_to_euler(filter.x))
            
            run_time.append(time2 - time1)

        x_array = np.array(x_list)[2:, :]
        y_array = np.array(y_list)[2:, :]
        x_hat_array = np.array(x_hat_list)[2:, :]

        rmse = np.mean(np.sqrt(np.mean((x_array -
                                         x_hat_array) ** 2, axis=(0))), axis=0)
        if rmse < 8:
            x_mc.append(x_array)
            y_mc.append(y_array)
            x_hat_mc.append(x_hat_array)
            all_time.append(np.mean(run_time))

    x_mc = np.array(x_mc)
    y_mc = np.array(y_mc)
    x_hat_mc = np.array(x_hat_mc)
    mean_time = np.mean(all_time)
    print(x_mc.shape)

    data_dict = {'x_mc': x_mc, 'y_mc': y_mc, 'x_hat_mc': x_hat_mc, 'mean_time': mean_time}
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


def quat_to_euler(q):
    w, x, y, z = q
    cos_pitch_cos_yaw = 1.0 - 2.0 * (y*y + z*z)
    cos_pitch_sin_yaw = 2.0 * (x*y + w*z) 
    sin_pitch = - 2.0 * (x*z - w*y) 
    cos_pitch = 0.0
    sin_roll_cos_pitch = 2.0 * (y*z + w*x)  
    cos_roll_cos_pitch = 1.0 - 2.0 * (x*x + y*y)

    cos_pitch = np.sqrt(cos_pitch_cos_yaw*cos_pitch_cos_yaw + cos_pitch_sin_yaw*cos_pitch_sin_yaw)
    yaw = np.arctan2(cos_pitch_sin_yaw, cos_pitch_cos_yaw)
    if abs(sin_pitch) >= 1:
        pitch = np.sign(sin_pitch) * np.pi / 2
    else:
        pitch = np.arcsin(sin_pitch)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    
    euler = np.array([roll, pitch, yaw])

    return euler
