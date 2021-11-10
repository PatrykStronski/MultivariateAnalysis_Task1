import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from norm_dist_models import std_normal, log_normal, gamma_norm, exp_norm, exponential
from scipy.stats import gaussian_kde 
from lib import non_paramteric_histogram, get_moments, draw_some_estimations

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

property = 'fire_size'

mean, median, var, skewness  = get_moments(df, property)
stddev = math.sqrt(var)
print(f'For {property} we have mean: {mean}; median: {median}; stddev: {stddev}')

kernel = gaussian_kde(df[property])
min_amount, max_amount = df[property].min(), df[property].max()
x = np.linspace(min_amount, max_amount, num=len(df))
kde_values = kernel(x)

sns.displot(data=df, x=property, label=f'Average size for ll classes', bins=500, stat="probability")
plt.plot(x, kde_values, label='KDE')
plt.plot(x, gamma_norm(x, mean, stddev), label='GAMMA')
plt.plot(x, std_normal(x, mean, stddev), label='normal')
plt.plot(x, log_normal(x, skewness, mean, stddev), label='lognormal')
plt.plot(x, exp_norm(x, mean, stddev), label='EXP normal')
plt.plot(x, exponential(x), label='EXPonential')
plt.legend()
plt.show()

plt.boxplot(df[property], vert=False)
plt.show()
