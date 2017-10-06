
""" Deep Learning Playground
"""
# Juan E Rolon 2017
#
# The present script implements a machine learning miniproject
# using a deep ('not so deep') learning approach using Keras.

# The goal of the miniproject is to predict student admissions to
# graduate school at UCLA based on three pieces GPA, GRE Scores and
# Class Rank

# The miniproject generalizes my scratch work towards completion
# of the Udacity Machine Learning Nanodegree
#
# Author: Juan E Rolon
# https://github.com/juanerolon/deep-learning-playground
#
# License: MIT

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

def inspect_dataset(data):
    """Prints data column headers (fields) and 5-row data preview"""
    print("\nData fields: {}".format(list(data.columns)))
    print("\nData preview:\n")
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
        #one-hot-encode y label
        y = np_utils.to_categorical(np.array(data[features]))
    elif as_numpy==False:
        X = df.drop(features, axis=1)
        # one-hot-encode y label
        y = pd.get_dummies(df[[features]], columns=[features])
    else:
        raise Exception('Incorrect as_numpy argument: {}'.format(as_numpy))

    return X, y

def build_inter_seq_model(input_dim, layer_loads, act_type):

    model = Sequential()
    model.add(Dense(layer_loads[0], input_dim=input_dim))
    model.add(Activation('sigmoid'))
    hidden_loads = layer_loads[1:]

    for load in hidden_loads:
        model.add(Dense(load))
        model.add(Activation(act_type))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



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
    #one-hot-encode rank feature
    proc_data = pd.get_dummies(data, columns=['rank'])
    X, y = split_data(proc_data, 'admit', as_numpy=True)

    print("Shape of X:", X.shape)
    print("\nShape of y:", y.shape)
    print("\nFirst 10 rows of X")
    print(X[:10])
    print("\nFirst 10 rows of y")
    print(y[:10])


    if True:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        model = build_inter_seq_model(input_dim=6, layer_loads= [128, 32, 2], act_type='sigmoid')
        print("\nModel Summary:\n")
        model.summary()

        print(X_train.shape)
        print(y_train.shape)

        model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=0)
        score = model.evaluate(X_test, y_test)

        print("Model Accuracy: {}".format(score))









