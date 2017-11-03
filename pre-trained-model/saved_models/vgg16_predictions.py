#@

#Illustrates model weights loading.
#We obtain weights from a vgg16 model saved from
#previous model fitting. Assumes known architecture.
#@juanerolon (Juan E. Rolon)
#https://github.com/juanerolon/


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
if False:
    config.gpu_options.allow_growth = True
    print("GPU memory incrementally allocated for current tensorflow session")

# Set to True if you decide to allocate a specific fraction of the total GPU
# memory to the current tensorflow session
if False:
    mem_frac = 0.7
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
if True:
    #ds_path = '/mnt/linuxdata2/Dropbox/_machine_learning/udacity_projects/cnn-project/'
    ds_path = '/Users/juanerolon/Dropbox/_machine_learning/udacity_projects/cnn-project/'
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




#*************************** PRE_TRAINED_MODEL ********************************************
#Load pre-trained features otained by a pre-trained VGG16 convolutional neural network

#Here we load the bottleneck features file from a different specified path different from
#ds_path defined above

features_path = '/Users/juanerolon/Dropbox/_machine_learning/ai/bottleneck_features/'

bottleneck_features = np.load(features_path + 'DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


#------------------------------- Known Model ----------------------------------

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))
VGG16_model.summary()

#----------- Load Weights from prior checkpointed training  -------------
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5') #assumes file in local directory
print("Created model and loaded weights from file")

#----------- Compile Model -------------
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]
# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#as process runs in background we need to close tensorflow session manually
tf.Session().close()

