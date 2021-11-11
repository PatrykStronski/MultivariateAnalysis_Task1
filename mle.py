import pandas as pd
import math
from scipy.optimize import minimize
from scipy.stats.distributions import norm, gamma, exponnorm, lognorm, expon

def gamma_log(x: pd.Series, alpha_beta: tuple) -> list:
    answs = []
    for g in gamma.pdf(x, alpha_beta[0], alpha_beta[1]):
        if g > 0:
            answs.append(math.log(g))
        else:
            answs.append(0)
    return answs

def mle_gamma(x: pd.Series, a: float, b: float) -> tuple:
    minimize_fx = lambda alpha_beta: sum(gamma_log(x, alpha_beta))
    answ = minimize(minimize_fx, (a, b), method='BFGS', tol=0.001)
    return answ.x

def gamma_lognorm(x: pd.Series, alpha_beta: tuple) -> list:
    answs = []
    for g in lognorm.pdf(x, alpha_beta[0], loc=alpha_beta[1], scale=alpha_beta[2]):
        if g > 0:
            answs.append(math.log(g))
        else:
            answs.append(0)
    return answs

def mle_lognormal(x: pd.Series, a: float, b: float, c: float) -> tuple:
    minimize_fx = lambda alpha_beta: sum(gamma_lognorm(x, alpha_beta))
    answ = minimize(minimize_fx, (a, b, c), method='BFGS', tol=0.001)
    return answ.x