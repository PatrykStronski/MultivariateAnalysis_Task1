import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.distributions import gamma, lognorm, cosine
import scipy
from lib import non_paramteric_histogram, get_mean_median_var, draw_some_estimations

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

fire_size_classes = pd.unique(df['fire_size_class'])
property = 'fire_size'

for s_c in fire_size_classes:
    df_sampled = df.loc[df['fire_size_class'] == s_c]

    mean, median, var  = get_mean_median_var(df_sampled, property)
    stddev = math.sqrt(var)
    print(f'For {property} {s_c} we have mean: {mean}; median: {median}; stddev: {stddev}')

    kernel = scipy.stats.gaussian_kde(df_sampled[property])
    min_amount, max_amount = df_sampled[property].min(), df_sampled[property].max()
    x = np.linspace(min_amount, max_amount, num=40)
    kde_values = kernel(x)
    alpha_mom = mean ** 2 / var
    beta_mom = var / mean

    delta = math.sqrt(math.log(var **2 / mean ** 2 + 1))
    mi = math.log(mean - math.sqrt(var **2 / mean ** 2 + 1))

    sns.displot(data=df_sampled, x=property, label=f'Average size for class {s_c}', bins=10, stat="probability")
    plt.plot(x, kde_values, color='red', label='KDE')
    plt.plot(x, gamma.pdf(x, alpha_mom, beta_mom), color='green', label='GAMMA')
    plt.plot(x, lognorm.pdf(x, delta, mi), color='yellow', label='lognorm')
    plt.plot(x, cosine.pdf(x), color='violet', label='cosine')
    plt.legend()
    plt.show()