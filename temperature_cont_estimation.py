import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from norm_dist_models import log_normal, gamma_norm, exp_norm, exponential
from scipy.stats import gaussian_kde 
from lib import get_moments
from scipy.stats.distributions import norm, gamma, exponnorm, lognorm, expon
from least_squares import ls_gamma, ls_lognorm, ls_exponnorm
from qq_plot import draw_qq
from statistical_tests import calculate_tests

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

fire_size_classes = pd.unique(df['fire_size_class'])
property = 'Temp_cont'

for s_c in fire_size_classes:
    df_sampled = df.loc[df['fire_size_class'] == s_c]

    mean, median, var, skewness  = get_moments(df_sampled, property)
    stddev = math.sqrt(var)
    print(f'For {property} {s_c} we have mean: {mean}; median: {median}; stddev: {stddev}')
    print(df_sampled.size)

    kernel = gaussian_kde(df_sampled[property])
    min_amount, max_amount = df_sampled[property].min(), df_sampled[property].max()
    x = np.linspace(0, max_amount, num=50)
    kde_values = kernel(x)

    sns.displot(data=df_sampled, x=property, label=f'Average size for class {s_c}', bins=50, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_norm(x, mean, stddev), label='GAMMA')
    plt.plot(x, log_normal(x, skewness, min_amount, stddev), label='lognormal')
    plt.plot(x, exp_norm(x, min_amount, stddev), label='EXP normal')
    plt.plot(x, exponential(x), label='EXPonential')
    plt.legend()
    plt.show()

    mle_gm = gamma.fit(df_sampled[property])
    mle_ln = lognorm.fit(df_sampled[property])
    mle_en = exponnorm.fit(df_sampled[property])

    sns.displot(data=df_sampled, x=property, label=f'Average size for {property} class', bins=50, stat="probability")
    plt.plot(x, gamma.pdf(x, mle_gm[0], mle_gm[1], mle_gm[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mle_ln[0], loc=mle_ln[1], scale=mle_ln[2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, mle_en[0], loc=mle_en[1], scale=mle_en[2]), label='exponnorm')
    plt.legend()
    plt.show()

    draw_qq(kde_values, gamma.pdf(x, mle_gm[0], mle_gm[1], mle_gm[2]), 'gamma')
    calculate_tests(kde_values, 'gamma', mle_gm)
    draw_qq(kde_values, lognorm.pdf(x, mle_ln[0], loc=mle_ln[1], scale=mle_ln[2]), 'lognorm')
    calculate_tests(kde_values, 'lognorm', mle_ln)
    draw_qq(kde_values, exponnorm.pdf(x, mle_en[0], loc=mle_en[1], scale=mle_en[2]), 'exponnorm')
    calculate_tests(kde_values, 'exponnorm', mle_en)

    ls_gm = ls_gamma(x, kde_values, (1, mean, stddev))
    ls_ln = ls_lognorm(x, kde_values, (1, mean, stddev))
    ls_en = ls_exponnorm(x, kde_values, (1, mean, stddev))

    sns.displot(data=df_sampled, x=property, label=f'Average size for {property} class', bins=50, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, ls_gm[0], ls_gm[1], ls_gm[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, ls_ln[0], loc=ls_ln[1], scale=ls_ln[2]), label='lognormal')
    plt.plot(x, exponnorm.pdf(x, ls_en[0], loc=ls_en[1], scale=ls_en[2]), label='exponnorm')
    plt.legend()
    plt.show()

    draw_qq(kde_values, gamma.pdf(x, ls_gm[0], ls_gm[1], ls_gm[2]), 'gamma')
    calculate_tests(kde_values, 'gamma', ls_gm)
    draw_qq(kde_values, lognorm.pdf(x, ls_ln[0], loc=ls_ln[1], scale=ls_ln[2]), 'lognorm')
    calculate_tests(kde_values, 'lognorm', ls_ln)
    draw_qq(kde_values, exponnorm.pdf(x, ls_en[0], loc=ls_en[1], scale=ls_en[2]), 'exponnorm')
    calculate_tests(kde_values, 'exponnorm', ls_en)

    plt.boxplot(df_sampled[property], vert=False)
    plt.show()

