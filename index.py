import math
import pandas as pd
from conf import VARIABLES_CONTINUOUS
from lib import non_paramteric_histogram, get_mean_median_var, draw_some_estimations

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

for property in VARIABLES_CONTINUOUS:
    mean, median, var  = get_mean_median_var(df, property)
    stddev = math.sqrt(var)
    print(f'For {property} we have mean: {mean}; median: {median}; stddev: {stddev}')
    non_paramteric_histogram(df, property)
    #draw_some_estimations(df, property, mean, var)