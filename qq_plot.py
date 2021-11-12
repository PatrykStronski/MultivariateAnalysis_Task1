import pandas as pd
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
import statsmodels.api as sm

def draw_qq(kde_values: pd.Series, estimated: pd.Series, title: str):
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

    plt.title(f'QQ-plot for {title}')
    plt.show()