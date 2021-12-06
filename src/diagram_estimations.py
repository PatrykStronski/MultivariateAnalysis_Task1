import math
from scipy.stats.distributions import gamma, exponnorm, lognorm, expon, norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.qq_plot import draw_qq
from src.statistical_tests import calculate_tests, calculate_tests_v2, choose_best_fitness
from src.least_squares import ls_gamma, ls_lognorm, ls_expon, ls_exponnorm, ls_normal

MAIN_FOLDER = './figures/'
DIAGRAM_FOLDER = './figures/diagrams/'

dstrs = {
    'gamma': gamma.pdf,
    'exponnorm': exponnorm.pdf,
    'lognorm': lognorm.pdf,
    'expon': expon.pdf,
    'norm': norm.pdf
}

def draw_hist_kde(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    sns.displot(data=df, x=property, label=f'Distribution of {property}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.title(f'Histogram and KDE for {property} for class {fire_size_class}')
    plt.legend()
    plt.savefig(f'{MAIN_FOLDER}kde_histogram_{property}_{fire_size_class}.png', bbox_inches='tight')
    plt.show()

def draw_mm_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mm = {}
    mm['gamma'] = gamma.fit(df[property], method='MM')
    mm['lognorm'] = lognorm.fit(df[property], method='MM')
    mm['exponnorm'] = exponnorm.fit(df[property], method='MM')
    mm['expon'] = expon.fit(df[property], method='MM')
    mm['norm'] = norm.fit(df[property], method='MM')

    gamma_est = gamma.pdf(x, mm['gamma'][0], mm['gamma'][1], mm['gamma'][2])
    lognorm_est = lognorm.pdf(x, mm['lognorm'][0], mm['lognorm'][1], mm['lognorm'][2])
    exponnorm_est = exponnorm.pdf(x, mm['exponnorm'][0], mm['exponnorm'][1], mm['exponnorm'][2])
    expon_est = expon.pdf(x, mm['expon'][0], mm['expon'][1])
    norm_est = norm.pdf(x, mm['norm'][0], mm['norm'][1])

    sns.displot(data=df, x=property, label=f'Distribution of {property}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_est, label='GAMMA')
    plt.plot(x, lognorm_est, label='lognormal')
    plt.plot(x, exponnorm_est, label='EXP normal')
    plt.plot(x, expon_est, label='EXPonential')
    plt.plot(x, norm_est, label='Normal')
    plt.title(f'Method Of Moments for variable {property}')
    plt.legend()
    plt.savefig(f'{DIAGRAM_FOLDER}method_moments_{property}_{fire_size_class}.png', bbox_inches='tight')
    plt.show()

    test_results = []
    test_results.append(calculate_tests_v2(kde_values, gamma_est, 'gamma'))
    test_results.append(calculate_tests_v2(kde_values, lognorm_est, 'lognorm'))
    test_results.append(calculate_tests_v2(kde_values, exponnorm_est, 'exponnorm'))
    test_results.append(calculate_tests_v2(kde_values, expon_est, 'expon'))
    test_results.append(calculate_tests_v2(kde_values, norm_est, 'norm'))

    draw_qq(kde_values, gamma.pdf(x, mm['gamma'][0], mm['gamma'][1], mm['gamma'][2]), 'gamma', 'method of moments')
    draw_qq(kde_values, lognorm.pdf(x, mm['lognorm'][0], mm['lognorm'][1], mm['lognorm'][2]), 'lognorm', 'method of moments')
    draw_qq(kde_values, exponnorm.pdf(x, mm['exponnorm'][0], mm['exponnorm'][1], mm['exponnorm'][2]), 'exponnorm', 'method of moments')
    draw_qq(kde_values, expon.pdf(x, mm['expon'][0], mm['expon'][1]), 'expon', 'method of moments')
    draw_qq(kde_values, norm.pdf(x, mm['norm'][0], mm['norm'][1]), 'norm', 'method of moments')


def draw_mle_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mle = {}
    mle['gamma'] = gamma.fit(df[property])
    mle['lognorm'] = lognorm.fit(df[property])
    mle['exponnorm'] = exponnorm.fit(df[property])
    mle['expon'] = expon.fit(df[property])
    mle['norm'] = norm.fit(df[property])

    gamma_est = gamma.pdf(x, mle['gamma'][0], mle['gamma'][1], mle['gamma'][2])
    lognorm_est = lognorm.pdf(x, mle['lognorm'][0], mle['lognorm'][1], mle['lognorm'][2])
    exponnorm_est = exponnorm.pdf(x, mle['exponnorm'][0], mle['exponnorm'][1], mle['exponnorm'][2])
    expon_est = expon.pdf(x, mle['expon'][0], mle['expon'][1])
    norm_est = norm.pdf(x, mle['norm'][0], mle['norm'][1])

    sns.displot(data=df, x=property, label=f'Distribution of {property}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_est, label='GAMMA')
    plt.plot(x, lognorm_est, label='lognormal')
    plt.plot(x, exponnorm_est, label='EXP normal')
    plt.plot(x, expon_est, label='EXPonential')
    plt.plot(x, norm_est, label='Normal')
    plt.title(f'Maximum Likelihood Estimation for variable {property}')
    plt.legend()
    plt.savefig(f'{DIAGRAM_FOLDER}maximum_likelihood_{property}_{fire_size_class}.png', bbox_inches='tight')
    plt.show()

    test_results = []
    test_results.append(calculate_tests_v2(kde_values, gamma_est, 'gamma'))
    test_results.append(calculate_tests_v2(kde_values, lognorm_est, 'lognorm'))
    test_results.append(calculate_tests_v2(kde_values, exponnorm_est, 'exponnorm'))
    test_results.append(calculate_tests_v2(kde_values, expon_est, 'expon'))
    test_results.append(calculate_tests_v2(kde_values, norm_est, 'norm'))

    draw_qq(kde_values, gamma.pdf(x, mle['gamma'][0], mle['gamma'][1], mle['gamma'][2]), 'gamma', 'MLE')
    draw_qq(kde_values, lognorm.pdf(x, mle['lognorm'][0], mle['lognorm'][1], mle['lognorm'][2]), 'lognorm', 'MLE')
    draw_qq(kde_values, exponnorm.pdf(x, mle['exponnorm'][0], mle['exponnorm'][1], mle['exponnorm'][2]), 'exponnorm', 'MLE')
    draw_qq(kde_values, expon.pdf(x, mle['expon'][0], mle['expon'][1]), 'expon', 'MLE')
    draw_qq(kde_values, norm.pdf(x, mle['norm'][0], mle['norm'][1]), 'norm', 'MLE')

def draw_ls_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mean = df[property].mean()
    stddev = math.sqrt(df[property].var())

    ls = {}
    ls['gamma'] = ls_gamma(x, kde_values, (1, mean, stddev))
    ls['lognorm'] = ls_lognorm(x, kde_values, (1, mean, stddev))
    ls['exponnorm'] = ls_exponnorm(x, kde_values, (1, mean, stddev))
    ls['expon'] = ls_expon(x, kde_values, (1, mean, stddev))
    ls['norm'] = ls_normal(x, kde_values, (1, mean, stddev))
    
    gamma_est = gamma.pdf(x, ls['gamma'][0], ls['gamma'][1], ls['gamma'][2])
    lognorm_est = lognorm.pdf(x, ls['lognorm'][0], ls['lognorm'][1], ls['lognorm'][2])
    exponnorm_est = exponnorm.pdf(x, ls['exponnorm'][0], ls['exponnorm'][1], ls['exponnorm'][2])
    expon_est = expon.pdf(x, ls['expon'][0], ls['expon'][1])
    norm_est = norm.pdf(x, ls['norm'][0], ls['norm'][1])

    sns.displot(data=df, x=property, label=f'Distribution of {property}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_est, label='GAMMA')
    plt.plot(x, lognorm_est, label='lognormal')
    plt.plot(x, exponnorm_est, label='EXP normal')
    plt.plot(x, expon_est, label='EXPonential')
    plt.plot(x, norm_est, label='Normal')
    plt.title(f'Least Squares Estimation for variable {property}')
    plt.legend()
    plt.savefig(f'{DIAGRAM_FOLDER}least_squares_{property}_{fire_size_class}.png', bbox_inches='tight')
    plt.show()

    test_results = []
    test_results.append(calculate_tests_v2(kde_values, gamma_est, 'gamma'))
    test_results.append(calculate_tests_v2(kde_values, lognorm_est, 'lognorm'))
    test_results.append(calculate_tests_v2(kde_values, exponnorm_est, 'exponnorm'))
    test_results.append(calculate_tests_v2(kde_values, expon_est, 'expon'))
    test_results.append(calculate_tests_v2(kde_values, norm_est, 'norm'))

    draw_qq(kde_values, gamma.pdf(x, ls['gamma'][0], ls['gamma'][1], ls['gamma'][2]), 'gamma', 'LS')
    draw_qq(kde_values, lognorm.pdf(x, ls['lognorm'][0], ls['lognorm'][1], ls['lognorm'][2]), 'lognorm', 'LS')
    draw_qq(kde_values, exponnorm.pdf(x, ls['exponnorm'][0], ls['exponnorm'][1], ls['exponnorm'][2]), 'exponnorm', 'LS')
    draw_qq(kde_values, expon.pdf(x, ls['expon'][0], ls['expon'][1]), 'expon', 'LS')
    draw_qq(kde_values, norm.pdf(x, ls['norm'][0], ls['norm'][1]), 'norm', 'LS')