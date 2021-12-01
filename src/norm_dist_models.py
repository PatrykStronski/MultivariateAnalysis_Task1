import pandas as pd
import math
from scipy.stats.distributions import norm, gamma, exponnorm, lognorm, expon

def std_normal(x: pd.Series, mean: float, stddev: float) -> pd.Series:
    return norm.pdf(x, loc=mean, scale=stddev)

def log_normal(x: pd.Series, s: float, loc: float, stddev: float) -> pd.Series:
    return lognorm.pdf(x, s, loc=loc, scale=stddev)

def gamma_norm(x: pd.Series, mean: float, stddev: float) -> pd.Series:
    alpha_mom = (mean ** 2) / (stddev ** 2)
    beta_mom = (stddev ** 2) / mean
    return gamma.pdf(x, alpha_mom, beta_mom)

def exp_norm(x: pd.Series, loc: float, stddev: float) -> pd.Series:
    return exponnorm.pdf(x, 1.0, loc=loc, scale=stddev)

def exponential(x: pd.Series) -> pd.Series:
    return expon.pdf(x)