
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils

from sklearn.preprocessing import Normalizer, MinMaxScaler

def inspect_dataset(data):
    """Prints data column headers (fields) and 5-row data preview"""
    print("Data fields: {}".format(list(data.columns)))
    print("Data preview:\n")
    print(data.head())

def normalize_data(df, features):

    for feat in features:
        df[feat] = df[feat] / df[feat].max()

def split_data(df, features, as_numpy):
    """Splits the N-feature data (frame) between N-1 feature subset X and a 1-feature label y
       Returns the split as a tuple X, y either is numpy arrays or pandas dataframes.
    """
    if as_numpy==True:
        X = df.drop(features, axis=1).values
        y = np_utils.to_categorical(np.array(data[features]))
    elif as_numpy==False:
        X = df.drop(features, axis=1)
        y = pd.get_dummies(df[[features]], columns=[features])
    else:
        raise Exception('Incorrect as_numpy argument: {}'.format(as_numpy))

    return X, y

#two-class scatter plot
def scatter_plot_grp(data, feature1, feature2, group_feature):
    """Generates scatter plot of two features field1, field2 grouped
    by another feature group_field
    """

    groups = data.groupby(group_feature)
    plt.xlabel(feature1)
    plt.ylabel(feature2)

    colors =['b','r','k','g']
    ct = 0
    for name, group in groups:
        plt.scatter(group.gre, group.gpa, c=colors[ct], label=name)
        ct+=1

    plt.legend()
    plt.show()

if __name__ == '__main__':


    #Set __location__ to current script location in file system
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    #Load dataset
    data = pd.read_csv(os.path.join(__location__, 'student_data.csv'))

    #Tests
    inspect_dataset(data)
    normalize_data(data, ['gpa', 'gre'])
    inspect_dataset(data)
    X, y = split_data(data, 'admit', as_numpy=True)

    print(X)
    print(y)







