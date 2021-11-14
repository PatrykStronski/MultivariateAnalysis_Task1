import math
from scipy.stats.distributions import gamma, exponnorm, lognorm, expon
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qq_plot import draw_qq
from statistical_tests import calculate_tests
from least_squares import ls_gamma, ls_lognorm, ls_exponnorm, ls_expon


def draw_mm_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mm_gamma = gamma.fit(df[property], method='MM')
    mm_lognorm = lognorm.fit(df[property], method='MM')
    mm_exponnorm = exponnorm.fit(df[property], method='MM')
    mm_expon = expon.fit(df[property], method='MM')

    sns.displot(data=df, x=property, label=f'Average size for class {fire_size_class}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, mm_gamma[0], mm_gamma[1], mm_gamma[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mm_lognorm[0], mm_lognorm[1], mm_lognorm[2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, mm_exponnorm[0], mm_exponnorm[1], mm_exponnorm[2]), label='EXP normal')
    plt.plot(x, expon.pdf(x, mm_expon[0], mm_expon[1]), label='EXPonential')
    plt.title(f'Method Of Moments from perspective of {property}')
    plt.legend()
    plt.show()

    draw_qq(kde_values, gamma.pdf(x, mm_gamma[0], mm_gamma[1], mm_gamma[2]), 'gamma')
    calculate_tests(kde_values, 'gamma', mm_gamma)
    draw_qq(kde_values, lognorm.pdf(x, mm_lognorm[0], loc=mm_lognorm[1], scale=mm_lognorm[2]), 'lognorm')
    calculate_tests(kde_values, 'lognorm', mm_lognorm)
    draw_qq(kde_values, exponnorm.pdf(x, mm_exponnorm[0], loc=mm_exponnorm[1], scale=mm_exponnorm[2]), 'exponnorm')
    calculate_tests(kde_values, 'exponnorm', mm_exponnorm)
    draw_qq(kde_values, expon.pdf(x, mm_expon[0], mm_expon[1]), 'expon')
    calculate_tests(kde_values, 'expon', mm_expon)


def draw_mle_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mle_gm = gamma.fit(df[property])
    mle_ln = lognorm.fit(df[property])
    mle_en = exponnorm.fit(df[property])
    mle_expon = expon.fit(df[property])

    sns.displot(data=df, x=property, label=f'Average size for {property} class', bins=binz, stat="probability")
    plt.plot(x, gamma.pdf(x, mle_gm[0], mle_gm[1], mle_gm[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mle_ln[0], mle_ln[1], mle_ln[2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, mle_en[0], mle_en[1], mle_en[2]), label='EXP normal')
    plt.plot(x, expon.pdf(x, mle_expon[0], mle_expon[1]), label='EXPonential')
    plt.title(f'Maximum Likelihood Estimation from perspective of {property}')
    plt.legend()
    plt.show()

    draw_qq(kde_values, gamma.pdf(x, mle_gm[0], mle_gm[1], mle_gm[2]), 'gamma')
    calculate_tests(kde_values, 'gamma', mle_gm)
    draw_qq(kde_values, lognorm.pdf(x, mle_ln[0], loc=mle_ln[1], scale=mle_ln[2]), 'lognorm')
    calculate_tests(kde_values, 'lognorm', mle_ln)
    draw_qq(kde_values, exponnorm.pdf(x, mle_en[0], loc=mle_en[1], scale=mle_en[2]), 'exponnorm')
    calculate_tests(kde_values, 'exponnorm', mle_en)
    draw_qq(kde_values, expon.pdf(x, mle_expon[0], mle_expon[1]), 'expon')
    calculate_tests(kde_values, 'expon', mle_expon)


def draw_ls_diagrams(df: pd.DataFrame, x: pd.Series, fire_size_class: str, property: str, kde_values: pd.Series, binz: int):
    mean = df[property].mean()
    stddev = math.sqrt(df[property].var())

    ls_gm = ls_gamma(x, kde_values, (1, mean, stddev))
    ls_ln = ls_lognorm(x, kde_values, (1, mean, stddev))
    ls_en = ls_exponnorm(x, kde_values, (1, mean, stddev))
    ls_ex = ls_expon(x, kde_values, (1, mean, stddev))

    sns.displot(data=df, x=property, label=f'Average size for {property} class {fire_size_class}', bins=binz, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, ls_gm[0], ls_gm[1], ls_gm[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, ls_ln[0], loc=ls_ln[1], scale=ls_ln[2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, ls_en[0], loc=ls_en[1], scale=ls_en[2]), label='exponnorm')
    plt.plot(x, expon.pdf(x, ls_ex[0], ls_ex[1]), label='EXPonential')
    plt.legend()
    plt.show()

    draw_qq(kde_values, gamma.pdf(x, ls_gm[0], ls_gm[1], ls_gm[2]), 'gamma')
    calculate_tests(kde_values, 'gamma', ls_gm)
    draw_qq(kde_values, lognorm.pdf(x, ls_ln[0], loc=ls_ln[1], scale=ls_ln[2]), 'lognorm')
    calculate_tests(kde_values, 'lognorm', ls_ln)
    draw_qq(kde_values, exponnorm.pdf(x, ls_en[0], loc=ls_en[1], scale=ls_en[2]), 'exponnorm')
    calculate_tests(kde_values, 'exponnorm', ls_en)
    draw_qq(kde_values, expon.pdf(x, ls_ex[0], ls_ex[1]), 'expon')
    calculate_tests(kde_values, 'expon', (ls_ex[0], ls_ex[1]))