import math
import numpy as np
from numpy.core.fromnumeric import std
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from norm_dist_models import std_normal, log_normal, gamma_norm, exp_norm, exponential
from scipy.stats import gaussian_kde 
from lib import non_paramteric_histogram, get_moments, draw_some_estimations
from scipy.stats.distributions import norm, gamma, exponnorm, lognorm, expon
from mle import mle_gamma, mle_lognormal

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
    x = np.linspace(min_amount, max_amount, num=50)
    kde_values = kernel(x)

    sns.displot(data=df_sampled, x=property, label=f'Average size for class {s_c}', bins=50, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_norm(x, mean, stddev), label='GAMMA')
    plt.plot(x, log_normal(x, skewness, min_amount, stddev), label='lognormal')
    plt.plot(x, exp_norm(x, min_amount, stddev), label='EXP normal')
    plt.plot(x, exponential(x), label='EXPonential')
    plt.legend()
    plt.show()

#    mle_gm = mle_gamma(x, mean, stddev)
#    mle_ln = mle_lognormal(x, skewness, mean, stddev)
    mle_gm = gamma.fit(df_sampled[property], loc=min_amount)
    #print(mle_gm)
    mle_ln = lognorm.fit(df_sampled[property], loc=min_amount)
    #print(mle_ln)

    sns.displot(data=df_sampled, x=property, label=f'Average size for {property} class', bins=50, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma.pdf(x, mle_gm[0], mle_gm[1], mle_gm[2]), label='GAMMA')
    plt.plot(x, lognorm.pdf(x, mle_ln[0], loc=mle_ln[1], scale=mle_ln[2]), label='lognormal')
    plt.legend()
    plt.show()

#    plt.boxplot(df_sampled[property], vert=False)
#    plt.show()
