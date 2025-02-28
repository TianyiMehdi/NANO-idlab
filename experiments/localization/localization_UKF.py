import sys
import argparse
import numpy as np

sys.path.append("./")
from experiments.save_and_plot import save_per_exp
from experiments.run import run_filter

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument("--filter_name", default="UKF", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="Localization", type=str, help="Name of the model")
    parser.add_argument("--noise_type", default="Beta", type=str, help="Name of the model")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--random_seed", default=42, type=int, help='Number of the random seed')
    parser.add_argument("--N_exp", default=50, type=int, help="Number of the MC experiments")
    parser.add_argument("--steps", default=200, type=int, help="Number of the steps in each trajectory")
    
    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Filter parameters
    parser_filter = argparse.ArgumentParser(description="filter_parameters")
    
    args_filter = parser_filter.parse_args()
    filter_dict = vars(args_filter)

    np.random.seed(args_dict['random_seed'])

    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []

    x_hat0 = np.array([0., 0., 0.])
    t = np.arange(1, args_dict['steps'] + 1)
    v_input = 5 * np.sin(2 * np.pi * t * 0.05 / 2)
    w_input = 3 * np.sin(2 * np.pi * t * 0.05 / 2)
    control_input = np.stack((v_input, w_input), axis=1)
    data_dict = run_filter(args_dict['N_exp'], args_dict['steps'], 
                           args_dict['model_name'], args_dict['noise_type'], 
                           args_dict['filter_name'], x_hat0, filter_dict,
                           control_input)

    save_per_exp(data_dict, args_dict, filter_dict)