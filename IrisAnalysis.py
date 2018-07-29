import numpy as np
import pandas as pd
import sklearn as sk
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 as sql
import sklearn as sk

def plot_width(df):
    param_x = []
    param_y = []
    param_x.append('PetalLengthCm')
    param_y.append('PetalWidthCm')
    param_x.append('SepalLengthCm')
    param_y.append('SepalWidthCm')

    for i in range(0, len(param_x)):
        fig = df.loc[df['Species'] == 'Iris-setosa'].plot.scatter(x=param_x[i], y=param_y[i], color='red',
                                                               label='Setosa')
        df.loc[df['Species'] == 'Iris-versicolor'].plot.scatter(x=param_x[i], y=param_y[i], color='blue',
                                                             label='versicolor', ax=fig)
        df.loc[df['Species'] == 'Iris-virginica'].plot.scatter(x=param_x[i], y=param_y[i], color='yellow',
                                                            label='virginica', ax=fig)
        fig.set_xlabel(param_x[i])
        fig.set_ylabel(param_y[i])
        fig.set_title(param_x[i] + ' vs. ' + param_y[i])
        plt.show()

def plot_distribution_flowers(df):
    df.hist()
    plt.show()


def main():
    path = './database.sqlite'
    conn = sql.connect(path)
    print(pd.read_sql('SELECT * FROM sqlite_master WHERE type="table";', conn))
    df = pd.read_sql('SELECT * FROM Iris;', conn)


    df = df.loc[(df['SepalLengthCm'].notnull()) & (df['SepalWidthCm'].notnull()) & (df['PetalLengthCm'].notnull()) &
    (df['PetalWidthCm'].notnull()) & (df['Species'].notnull())]

    print(df.head())
    #plot_width(df)
    #plot_distribution_flowers(df)

    plt.figure(figsize=(7, 4))
    sns.heatmap(df.corr())
    plt.show()



main()