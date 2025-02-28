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
    parser.add_argument("--filter_name", default="NANO", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="SequenceForcasting", type=str, help="Name of the model")
    parser.add_argument("--noise_type", default="Beta", type=str, help="Name of the model")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--random_seed", default=42, type=int, help='Number of the random seed')
    parser.add_argument("--N_exp", default=50, type=int, help="Number of the MC experiments")
    parser.add_argument("--steps", default=500, type=int, help="Number of the steps in each trajectory")
    
    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)

    parser_filter = argparse.ArgumentParser(description="filter_parameters")
    parser_filter.add_argument("--n_iterations", default=1, type=int, help="Iterations for NANO")
    parser_filter.add_argument("--lr", default=0.01, type=float, help="Learning Rate for NANO")
    parser_filter.add_argument("--init_type", default='iekf', type=str, help="Initialization type for Natural Gradient iteration, 'prior', 'laplace', 'iekf', 'ukf'")
    # Gaussian, Laplace : 5, 0.5, iekf, 1
    # Beta: 
    
    args_filter = parser_filter.parse_args()
    filter_dict = vars(args_filter)
    
    if filter_dict['init_type'] == 'iekf':
        parser_filter.add_argument("--iekf_max_iter", default=1, type=float, help="Iterations for iekf init")

    args_filter = parser_filter.parse_args()
    filter_dict = vars(args_filter)

    np.random.seed(args_dict['random_seed'])

    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []

    x_hat0 = np.array([-2.0, -2.0])
    data_dict = run_filter(args_dict['N_exp'], args_dict['steps'], args_dict['model_name'], args_dict['noise_type'], args_dict['filter_name'], x_hat0, filter_dict)

    save_per_exp(data_dict, args_dict, filter_dict)