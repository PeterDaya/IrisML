import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 as sql
from sklearn.cross_validation import train_test_split
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

def do_KNeighbors(x_train, y_train, x_test, y_test):
    kn = KNeighborsClassifier(n_neighbors=3)
    kn.fit(x_train, y_train)
    pred = kn.predict(x_test)
    acc = sk.metrics.accuracy_score(pred, y_test)
    acc = "%.5f" % (acc*100.0)
    print("Accuracy of KNN model: ", acc + "%")

def do_DecisionTree(x_train, y_train, x_test, y_test):
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    acc = sk.metrics.accuracy_score(pred, y_test)
    acc = "%.5f" % (acc * 100.0)
    print("Accuracy of Decision Tree model: ", acc + "%")


def main():
    path = './database.sqlite'
    conn = sql.connect(path)
    print(pd.read_sql('SELECT * FROM sqlite_master WHERE type="table";', conn))
    df = pd.read_sql('SELECT * FROM Iris;', conn)


    df = df.loc[(df['SepalLengthCm'].notnull()) & (df['SepalWidthCm'].notnull()) & (df['PetalLengthCm'].notnull()) &
    (df['PetalWidthCm'].notnull()) & (df['Species'].notnull())]

    print(df.head())
    ''' 
    plot_width(df)
    plot_distribution_flowers(df)

    plt.figure(figsize=(7, 4))
    sns.heatmap(df.corr())
    plt.show()
    '''

    dftrain, dftest = train_test_split(df, test_size=0.3)
    x_train = dftrain.loc[:, (df.columns != 'Id') & (df.columns != 'Species')]
    x_test = dftest.loc[:, (df.columns != 'Id') & (df.columns != 'Species')]
    y_train = dftrain['Species']
    y_test = dftest['Species']
    print(x_train)
    print(x_test)

    do_KNeighbors(x_train, y_train, x_test, y_test)
    do_DecisionTree(x_train, y_train, x_test, y_test)



main()