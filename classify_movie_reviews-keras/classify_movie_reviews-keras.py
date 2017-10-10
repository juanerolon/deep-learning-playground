""" Deep Learning Playground
"""
# Juan E Rolon 2017
#
# The present script implements a machine learning miniproject
# using a deep ('not so deep') learning approach using Keras.

# The goal of the miniproject is to classify the degreee of positivity of
# movie reviews from imdb database according to the most frequent words
# appearing in the review text

# The miniproject generalizes my scratch work towards completion
# of the Udacity Machine Learning Nanodegree
#
# Author: Juan E Rolon
# https://github.com/juanerolon/deep-learning-playground
#
# License: MIT:


#The imdb data comes preloaded in Keras, which means we don't need to open or read any files manually.

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)


print("\nData Preview - Raw Features Training Set:\n")
for row in x_train[0:5]:
    print(row)

print("\nData Preview - Raw Labels Training Set:\n")

for row in y_train[0:5]:
    print(row)

print("\nData Sets Shapes:\n")
print("Training features set X has shape {}".format(x_train.shape))
print("Training labels set y has shape {}".format(y_train.shape))
print("Testing features set X has shape {}".format(x_test.shape))
print("Testing labels set y has shape {}".format(y_test.shape))

from keras.preprocessing.text import Tokenizer
#Tokenizer is a class for vectorizing texts, or/and turning texts into sequences
#(=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i)


#Token vectors contain 1000 elements
tokenizer = Tokenizer(num_words=1000)

#Vectorize training and test set according to the tokenizer specifications
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

print("\nTraining Feature Data Sets Shapes After Tokenizer Vectorization:\n")
print("Training features set X has shape {}".format(x_train.shape))
print("Testing features set X has shape {}".format(x_test.shape))

# One-hot encoding the output labels for both training and testing sets
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("\nTraining and Testing Label Sets Shapes After One-Hot-Encoding:\n")
print("Training one-hot-encoded labels set y has shape {}".format(y_train.shape))
print("Testing one-hot-encoded labels set y has shape {}".format(y_test.shape))