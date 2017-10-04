
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python.control_flow_ops = tf

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# Set random seed
np.random.seed(42)

# Our data
#input of feature data (boolean tuples or tuples bits)
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')

#output labels y (boolean, logical bit state)
y = np.array([[0],[1],[1],[0]]).astype('float32')

#Test structure and shape of X and y arrays
if True:
    print("X array of input features with shape {}\n".format(X.shape))
    print(X, "\n")
    print("y array of output labels with shape {}\n".format(y.shape))
    print(y, "\n")


# One-hot encoding the output
y = np_utils.to_categorical(y)

if True:
    print("y array of one-hot enconded labels with shape {}\n".format(y.shape))
    print(y, "\n")

# Building the model
# Define xor as the object resulting from instantiating the Sequential class (Sequential model)
xor = Sequential()

# Implement the add method as neccessary to add the different layers to the neural network
#
#Create 32 nodes which each expect to receive 2-element vectors as inputs
xor.add(Dense(32, input_dim=2))
xor.add(Activation("sigmoid"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))

#Compile code in the back-end of Tensorflow
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

# Print the model architecture
print("Model architecture:\n")
xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=1000, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))