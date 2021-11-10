import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from norm_dist_models import std_normal, log_normal, gamma_norm, exp_norm, exponential
from scipy.stats import gaussian_kde 
from lib import non_paramteric_histogram, get_moments, draw_some_estimations

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

fire_size_classes = pd.unique(df['fire_size_class'])
fire_size_classes = []
property = 'fire_size'

for s_c in fire_size_classes:
    df_sampled = df.loc[df['fire_size_class'] == s_c]

    mean, median, var, skewness  = get_moments(df_sampled, property)
    stddev = math.sqrt(var)
    print(f'For {property} {s_c} we have mean: {mean}; median: {median}; stddev: {stddev}')

    kernel = gaussian_kde(df_sampled[property])
    min_amount, max_amount = df_sampled[property].min(), df_sampled[property].max()
    x = np.linspace(min_amount, max_amount, num=len(df_sampled))
    kde_values = kernel(x)

    sns.displot(data=df_sampled, x=property, label=f'Average size for class {s_c}', bins=50, stat="probability")
    plt.plot(x, kde_values, label='KDE')
    plt.plot(x, gamma_norm(x, mean, stddev), label='GAMMA')
    plt.plot(x, std_normal(x, mean, stddev), label='normal')
    plt.plot(x, log_normal(x, skewness, mean, stddev), label='lognormal')
    plt.plot(x, exp_norm(x, mean, stddev), label='EXP normal')
    plt.plot(x, exponential(x), label='EXPonential')
    plt.legend()
    plt.show()

    plt.boxplot(df_sampled[property], vert=False)
    plt.show()
