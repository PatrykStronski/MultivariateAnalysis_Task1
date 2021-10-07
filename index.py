import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_diagrams(data_frame: pd.DataFrame):
    plt.hist(data_frame.x, bins=20, label='x-axis')
    plt.hist(data_frame.y, bins=20, label='y-axis')
    plt.hist(data_frame.s, bins=20, label='speed')
    plt.show()

    plt.hist2d(data_frame.x, data_frame.y, bins=80)
    plt.show()


pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 25)

df = pd.read_csv('./data/tracking2018.csv')

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
