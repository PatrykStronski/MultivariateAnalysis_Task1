import math
from scipy.stats.distributions import gamma, exponnorm, lognorm, expon
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qq_plot import draw_qq
from statistical_tests import calculate_tests, choose_best_fitness
from least_squares import ls_gamma, ls_lognorm, ls_expon, ls_exponnorm

dstrs = {
    'gamma': gamma.pdf,
    'exponnorm': exponnorm.pdf,
    'lognorm': lognorm.pdf,
    'expon': expon.pdf
}

def draw_mm_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mm = {}
    mm['gamma'] = gamma.fit(df[property], method='MM')
    mm['lognorm'] = lognorm.fit(df[property], method='MM')
    mm['exponnorm'] = exponnorm.fit(df[property], method='MM')
    mm['expon'] = expon.fit(df[property], method='MM')

    sns.displot(data=df, x=property, label=f'Average size for class {fire_size_class}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, mm['gamma'][0], mm['gamma'][1], mm['gamma'][2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mm['lognorm'][0], mm['lognorm'][1], mm['lognorm'][2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, mm['exponnorm'][0], mm['exponnorm'][1], mm['exponnorm'][2]), label='EXP normal')
    plt.plot(x, expon.pdf(x, mm['expon'][0], mm['expon'][1]), label='EXPonential')
    plt.title(f'Method Of Moments from perspective of {property}')
    plt.legend()
    plt.show()

    test_results = []
    test_results.append(calculate_tests(kde_values, 'gamma', mm['gamma']))
    test_results.append(calculate_tests(kde_values, 'lognorm', mm['lognorm']))
    test_results.append(calculate_tests(kde_values, 'exponnorm', mm['exponnorm']))
    test_results.append(calculate_tests(kde_values, 'expon', mm['expon']))

    best_fit_ks, best_fit_omega = choose_best_fitness(test_results)
    if best_fit_ks['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_ks['dist']](x, mm[best_fit_ks['dist']][0], mm[best_fit_ks['dist']][1], mm[best_fit_ks['dist']][2]), 'gamma')
    if best_fit_omega['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_omega['dist']](x, mm[best_fit_omega['dist']][0], mm[best_fit_omega['dist']][1], mm[best_fit_omega['dist']][2]), 'gamma')


def draw_mle_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mle = {}
    mle['gamma'] = gamma.fit(df[property])
    mle['lognorm'] = lognorm.fit(df[property])
    mle['exponnorm'] = exponnorm.fit(df[property])
    mle['expon'] = expon.fit(df[property])

    sns.displot(data=df, x=property, label=f'Average size for {property} class {fire_size_class}', bins=binz, stat="probability")
    plt.plot(x, gamma.pdf(x, mle['gamma'][0], mle['gamma'][1], mle['gamma'][2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mle['lognorm'][0], mle['lognorm'][1], mle['lognorm'][2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, mle['exponnorm'][0], mle['exponnorm'][1], mle['exponnorm'][2]), label='EXP normal')
    plt.plot(x, expon.pdf(x, mle['expon'][0], mle['expon'][1]), label='EXPonential')
    plt.title(f'Maximum Likelihood Estimation from perspective of {property}')
    plt.legend()
    plt.show()

    test_results = []
    test_results.append(calculate_tests(kde_values, 'gamma', mle['gamma']))
    test_results.append(calculate_tests(kde_values, 'lognorm', mle['lognorm']))
    test_results.append(calculate_tests(kde_values, 'exponnorm', mle['exponnorm']))
    test_results.append(calculate_tests(kde_values, 'expon', mle['expon']))

    best_fit_ks, best_fit_omega = choose_best_fitness(test_results)
    if best_fit_ks['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_ks['dist']](x, mle[best_fit_ks['dist']][0], mle[best_fit_ks['dist']][1], mle[best_fit_ks['dist']][2]), 'gamma')
    if best_fit_omega['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_omega['dist']](x, mle[best_fit_omega['dist']][0], mle[best_fit_omega['dist']][1], mle[best_fit_omega['dist']][2]), 'gamma')
    

def draw_ls_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mean = df[property].mean()
    stddev = math.sqrt(df[property].var())

    ls = {}
    ls['gamma'] = ls_gamma(x, kde_values, (1, mean, stddev))
    ls['lognorm'] = ls_lognorm(x, kde_values, (1, mean, stddev))
    ls['exponnorm'] = ls_exponnorm(x, kde_values, (1, mean, stddev))
    ls['expon'] = ls_expon(x, kde_values, (1, mean, stddev))

    sns.displot(data=df, x=property, label=f'Average size for {property} class {fire_size_class}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, ls['gamma'][0], ls['gamma'][1], ls['gamma'][2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, ls['lognorm'][0], loc=ls['lognorm'][1], scale=ls['lognorm'][2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, ls['exponnorm'][0], loc=ls['exponnorm'][1], scale=ls['exponnorm'][2]), label='exponnorm')
    plt.plot(x, expon.pdf(x, ls['expon'][0], ls['expon'][1]), label='EXPonential')
    plt.legend()
    plt.show()

    test_results = []
    test_results.append(calculate_tests(kde_values, 'gamma', ls['gamma']))
    test_results.append(calculate_tests(kde_values, 'lognorm', ls['lognorm']))
    test_results.append(calculate_tests(kde_values, 'exponnorm', ls['exponnorm']))
    test_results.append(calculate_tests(kde_values, 'expon', (ls['expon'][0], ls['expon'][1])))

    best_fit_ks, best_fit_omega = choose_best_fitness(test_results)
    if best_fit_ks['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_ks['dist']](x, ls[best_fit_ks['dist']][0], ls[best_fit_ks['dist']][1], ls[best_fit_ks['dist']][2]), 'gamma')
    if best_fit_omega['dist'] != None:
        draw_qq(kde_values, dstrs[best_fit_omega['dist']](x, ls[best_fit_omega['dist']][0], ls[best_fit_omega['dist']][1], ls[best_fit_omega['dist']][2]), 'gamma')
    