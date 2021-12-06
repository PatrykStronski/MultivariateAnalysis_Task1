import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde 
from src.diagram_estimations import draw_mm_diagrams, draw_mle_diagrams, draw_ls_diagrams

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

fire_size_classes = pd.unique(df['fire_size_class'])
property = 'fire_size'

for s_c in fire_size_classes:
    df_sampled = df.loc[df['fire_size_class'] == s_c]

    kernel = gaussian_kde(df_sampled[property])
    min_amount, max_amount = df_sampled[property].min(), df_sampled[property].max()
    x = np.linspace(0, max_amount, num=50)
    kde_values = kernel(x)

    draw_mm_diagrams(df_sampled, x, s_c, property, kde_values, 50)

    draw_mle_diagrams(df_sampled, x, s_c, property, kde_values, 50)

    draw_ls_diagrams(df_sampled, x, s_c, property, kde_values, 50)

    plt.boxplot(df_sampled[property], vert=False)
    plt.show()
