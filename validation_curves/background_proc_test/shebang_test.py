import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
h = model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=10, verbose=1)
history_data = pd.DataFrame(h.history)

print('Model Performance Metrics vs Epoch Number:\n')
print(history_data.head())

#Save history to CSV file
history_data.to_csv('toy_keras_nn.csv')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('performance_metrics.pdf')

plt.figure(1,figsize=(11, 4))


plt.subplot(1,2,1)
# summarize history for accuracy
plt.plot(h.history['acc'],color='b')
plt.plot(h.history['val_acc'],color='k')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(1,2,2)
# summarize history for loss
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig(pp, format='pdf')
pp.close()


