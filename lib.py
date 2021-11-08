import scipy
import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

sns.set()

def get_mean_median_var(df: pd.DataFrame, property: str) -> (float, float, float):
    mean = df[property].mean()
    median = df[property].median()
    var = math.sqrt(df[property].var())
    return mean, median, var

def plot_data(df: pd.DataFrame, property: str):
    sns.histplot(df, x=property, bins=30)
    plt.show()

def non_paramteric_histogram(df: pd.DataFrame, property: str):
    kernel = scipy.stats.gaussian_kde(df[property])
    min_amount, max_amount = df[property].min(), df[property].max()
    x = np.linspace(min_amount, max_amount, num=40)
    kde_values = kernel(x)

    sns.displot(data=df, x=property, label=f'Average THIS', bins=40, stat="density")
    plt.plot(x, kde_values, color='red')
    plt.show()

def draw_some_estimations(df: pd.DataFrame, property: str, mean: float, var: float):
    alpha_mom = mean ** 2 / var
    beta_mom = var / mean

    min_amount, max_amount = df[property].min(), df[property].max()
    x = np.linspace(min_amount, max_amount, num=40)

    sns.displot(data=df, x=property, label=f'Average THIS', bins=40, stat="density")
    plt.plot(x, gamma.pdf(x, alpha_mom, beta_mom), color='green')
    plt.show()
