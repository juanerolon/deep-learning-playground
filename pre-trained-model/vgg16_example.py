#@juanerolon (Juan E. Rolon)
#https://github.com/juanerolon/deep-learning-playground

#needed for code execution timing
import time
import os

start_time = time.time()

##############################################
#Use tensfor flow memory management utilities
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()

# Set to True to allow GPU dynamic memory incremental allocation
# Note: memory is not deallocated automatically
if True:
    config.gpu_options.allow_growth = True
    print("GPU memory incrementally allocated for current tensorflow session")

# Set to True if you decide to allocate a specific fraction of the total GPU
# memory to the current tensorflow session
if False:
    mem_frac = 0.3
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    print("GPU memory allocated for current tensorflow session = {}".format(mem_frac))

set_session(tf.Session(config=config))

# Note: Use the NVIDIA System Management Interface to monitor your gpu compute devices
#      memory, e.g. $nvdia-smi from bash or using the cell below



#############################################
#load dog image datasets
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# Set to True specify path to folder containing dataset subfolders
# Note: useful if jupyter notebook not located in default location
# The location to path cnn-project changed due to corrupt repository on local premises
if True:
    ds_path = '/mnt/linuxdata2/Dropbox/_machine_learning/udacity_projects/cnn-project/'
else:
    ds_path = ''

# load train, test, and validation datasets
train_files, train_targets = load_dataset(ds_path + 'dogImages/train')
valid_files, valid_targets = load_dataset(ds_path + 'dogImages/valid')
test_files, test_targets = load_dataset(ds_path + 'dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob(ds_path + "dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.' % len(test_files))

#############################################
#load human image datasets
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob(ds_path + "lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


#############################################
#face detectors

import cv2
def face_detector(img_path):
    """returns "True" if face is detected in image stored at img_path"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


from keras.applications.resnet50 import ResNet50
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    """returns prediction vector for image located at img_path"""
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """returns "True" if a dog is detected in the image stored at img_path"""
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))



###########################################################
#image dataset pre-processing

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


#Implements Image Aumentation as done in aind2-cnn jupyter notebook for cifar10
#See https://github.com/udacity/aind2-cnn/blob/master/cifar10-augmentation/

from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# create and configure augmented image generator
datagen_valid = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(train_tensors)
datagen_valid.fit(valid_tensors)



#Load pre-trained features otained by a pre-trained VGG16 convolutional neural network
bottleneck_features = np.load(ds_path + 'bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


#Quick model build-up: join the pre-trained VGG16 feature layer wit the output layer
from keras.layers import GlobalAveragePooling2D
from keras.layers import  Dense
from keras.models import Sequential

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))
VGG16_model.summary()

#Compile pre-trained model
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#Check whether the folder for saving the checkpointer model weights
# e.g. saved_models folder, see below.
current_dir = os.path.dirname(os.path.realpath(__file__))
if not os.path.isdir(current_dir + "/saved_models"):
    os.system('mkdir saved_models')
else:
    pass


#Fit or 'train' model
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

#specify number of epochs and batch_size
epochs = 300
batch_size = 20


init_train_time = time.time()

# Set to True to fit the model to the non-augmented datasets
if False:
    h = VGG16_model.fit(train_VGG16, train_targets,
                    validation_data=(valid_VGG16, valid_targets),
                    epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

# Set to True to fit the model to the augmented datasets
if True:
    h = VGG16_model.fit_generator(datagen_train.flow(train_VGG16, train_targets, batch_size=batch_size),
                        steps_per_epoch=train_VGG16.shape[0] // batch_size,
                        epochs=epochs, verbose=1, callbacks=[checkpointer],
                        validation_data=datagen_valid.flow(valid_VGG16, valid_targets, batch_size=batch_size),
                        validation_steps=valid_VGG16.shape[0] // batch_size)

end_train_time = time.time()
tot_train_time = end_train_time-init_train_time

#Load optimized weights saved during training
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

###########################################################
#Save performance metrics to a file and pdf gigures

import pandas as pd
import matplotlib.pyplot as plt

#allows to store plt plots in files while script runs in background
plt.switch_backend('agg')


#Save history to CSV file
history_data = pd.DataFrame(h.history)
history_data.to_csv('performance_metrics.csv')

plt.figure(1, figsize=(10, 4))

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

plt.savefig('performance_metrics.png', dpi=300, orientation='landscape')

end_time = time.time()
exc_time =end_time-start_time

print("\n -------END---------- \n")
print("Training time = {0:.3f} minutes ".format(round(tot_train_time/60.0, 3)))
print("Total run time = {0:.3f} minutes".format(round(exc_time/60.0, 3)))

#as process runs in background we need to close tensorflow session manually
tf.Session().close()

