import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_diagrams(data_frame: pd.DataFrame):
    plt.hist(data_frame.x, bins=20, label='x-axis')
    plt.hist(data_frame.y, bins=20, label='y-axis')
    plt.hist(data_frame.s, bins=20, label='speed')
    plt.hist(data_frame.a, bins=20, label='acceleration')
    plt.show()

    plt.hist2d(data_frame.x, data_frame.y, bins=80)
    plt.show()
    plt.plot(data_frame.x, data_frame.y)
    plt.show()

def get_moments(data_frame: pd.DataFrame):
    mean_x = data_frame.x.mean()
    var_x = data_frame.x.var()

    mean_y = data_frame.y.mean()
    var_y = data_frame.y.var()

    mean_s = data_frame.s.mean()
    var_s = data_frame.s.var()

    return {
        mean_x,
        mean_y,
        mean_s,
        var_x,
        var_y,
        var_s
    }


pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 25)

df = pd.read_csv('./data/tracking2018.csv')
moments_matrix = pd.DataFrame(columns=['gameId', 'nflId', ])

players = df.nflId.unique()
games = df.gameId.unique()

for gameId in games:
    if pd.isna(gameId):
        continue

    for playerId in players:
        if pd.isna(playerId):
            continue

        previewData = df.loc[(df.nflId == playerId) & (df.gameId == gameId)]
        draw_diagrams(previewData)
        print_moments(previewData)
