import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import statsmodels.api as sm

QQPLOT_FOLDER = './figures/qqplots/'

def draw_qq(kde_values: pd.Series, estimated: pd.Series, distr: str, method: str):
    min_qn = np.min([kde_values.min(), estimated.min()])
    max_qn = np.max([kde_values.max(), estimated.max()])
    x = np.linspace(min_qn, max_qn)

    plt.plot(kde_values, estimated, ls="", marker="o", markersize=6)
    plt.plot(x, x, color="k", ls="--")
    plt.xlabel(f'KDE')
    plt.ylabel(f'Estimant')
    plt.xlim([min_qn, max_qn])
    plt.ylim([min_qn, max_qn])
    plt.grid(True)

    plt.title(f'QQ-plot for {distr} using {method}')
    plt.savefig(f'{QQPLOT_FOLDER}qqplot_{distr}_{method}.png', bbox_inches='tight')
    plt.show()