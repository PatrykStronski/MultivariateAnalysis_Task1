import pandas as pd
from typing import Callable
from scipy.optimize import least_squares
from scipy.stats.distributions import gamma, lognorm, exponnorm

def gamma_est(x: pd.Series, kde_values: pd.Series, args: tuple) -> list:
    g_pdf = gamma.pdf(x, args[0], args[1], args[2])
    return [kde_values[xn] - g_pdf[xn] for xn in range(0, len(x))]

def ls_gamma(x: pd.Series, kde_values: pd.Series, x0: tuple) -> tuple:
    est = lambda params: gamma_est(x, kde_values, params)
    answ = least_squares(est, x0)
    return answ.x

def lognorm_est(x: pd.Series, kde_values: pd.Series, args: tuple) -> list:
    g_pdf = lognorm.pdf(x, args[0], args[1], args[2])
    return [kde_values[xn] - g_pdf[xn] for xn in range(0, len(x))]

def ls_lognorm(x: pd.Series, kde_values: pd.Series, x0: tuple) -> tuple:
    est = lambda params: lognorm_est(x, kde_values, params)
    answ = least_squares(est, x0)
    return answ.x

def exponnorm_est(x: pd.Series, kde_values: pd.Series, args: tuple) -> list:
    g_pdf = exponnorm.pdf(x, args[0], args[1], args[2])
    return [kde_values[xn] - g_pdf[xn] for xn in range(0, len(x))]

def ls_exponnorm(x: pd.Series, kde_values: pd.Series, x0: tuple) -> tuple:
    est = lambda params: exponnorm_est(x, kde_values, params)
    answ = least_squares(est, x0)
    return answ.x