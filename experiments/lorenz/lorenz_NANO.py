import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.save_and_plot import save_per_exp
from experiments.run import run_filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--filter_name", default="NANO", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="Lorenz", type=str, help="Name of the model")
    parser.add_argument("--noise_type", default="Gaussian", type=str, help="Noise model type")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    parser.add_argument("--N_exp", default=50, type=int, help="Number of MC experiments")
    parser.add_argument("--steps", default=200, type=int, help="Number of steps per trajectory")

    args, remaining = parser.parse_known_args()
    args_dict = vars(args)

    parser_filter = argparse.ArgumentParser(description="filter_parameters")
    parser_filter.add_argument("--n_iterations", default=1, type=int, help="Iterations for NANO")
    parser_filter.add_argument("--lr", default=1, type=float, help="Learning rate for NANO")
    parser_filter.add_argument(
        "--init_type",
        default="ukf",
        type=str,
        choices=["prior", "laplace", "iekf", "ukf"],
        help="Initialization mode for NANO ('prior', 'laplace', 'iekf', 'ukf')",
    )

    args_filter, remaining = parser_filter.parse_known_args(remaining)
    filter_dict = vars(args_filter)

    if filter_dict["init_type"] == "iekf":
        parser_filter.add_argument("--iekf_max_iter", default=1, type=float, help="Iterations for iekf init")

    args_filter = parser_filter.parse_args(remaining)
    filter_dict = vars(args_filter)

    np.random.seed(args_dict["random_seed"])

    x_hat0 = np.array([1.0, 1.0, 1.0])
    data_dict = run_filter(
        args_dict["N_exp"],
        args_dict["steps"],
        args_dict["model_name"],
        args_dict["noise_type"],
        args_dict["filter_name"],
        x_hat0,
        filter_dict,
    )

    save_per_exp(data_dict, args_dict, filter_dict)
