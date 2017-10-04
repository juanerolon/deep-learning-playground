
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

def setModel(num_nodes=12, feat_dim=8, initializer = 'random_uniform'):

    """
    This function returns a deafault sequential model with a single layer with 12 artificial neurons,
    and it expects 8 input variables (also known as features). A sequential model is a linear pipeline
    or stack of neuron layers. The kernel initialization defines the way to set the initial random weights
    of layers e.g. 'random uniform', 'random normal', 'zero'. The neural network is Dense, so each neuron
    in a layer is connected to all neurons located in the previous layer and to all the neurons in the
    following layer.
    """
    model = Sequential()
    model.add(Dense(num_nodes, input_dim = feat_dim, kernel_initializer = initializer))
    return model

model = setModel(12,8)
model.summary()
