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
    parser = argparse.ArgumentParser(description="Lorenz coupling experiment with UKF.")
    parser.add_argument("--filter_name", default="UKF", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="LorenzCoupling", type=str, help="Name of the model")
    parser.add_argument("--noise_type", default="Gaussian", type=str, help="Noise model type")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    parser.add_argument("--N_exp", default=50, type=int, help="Number of MC experiments")
    parser.add_argument("--steps", default=200, type=int, help="Number of steps per trajectory")

    args = parser.parse_args()
    args_dict = vars(args)

    filter_dict = {}

    np.random.seed(args_dict["random_seed"])

    x_hat0 = np.zeros(9)
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
