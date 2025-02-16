import numpy as np

def get_beta_mean(alpha, beta):
    return alpha / (alpha + beta)

def get_beta_cov(alpha, beta):
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))