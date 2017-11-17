# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:00:42 2017

@author: AkshayJk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:37:09 2017

@author: AkshayJk
"""

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import EigenvalueRegularizer
from numpy.random import permutation
from keras.optimizers import SGD
import pandas as pd
import datetime
import glob
import cv2
import math
import pickle
from collections import OrderedDict
from keras import backend as K


vgg_weights = 'vgg16_weights.h5'
top_model_weights = 'fc_model.h5'
train_data_dir = 'train'
test_data_dir = 'test'

test_images_path = 'test/test'

img_width, img_height = 224, 224
nb_train_samples = 10194
nb_test_samples = 3354
color_type_global = 3

batch_size = 64

nb_epoch = 64

whole_weights = 'whole_model.h5'

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

assert os.path.exists(vgg_weights), 'Model weights not found (see "vgg_weights" variable in script).'
f = h5py.File(vgg_weights)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()

print('Model loaded.')

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(64, activation='relu', W_regularizer=EigenvalueRegularizer(6)))
top_model.add(Dense(6, activation='softmax', W_regularizer=EigenvalueRegularizer(6)))
#top_model.load_weights(top_model_weights)

model.add(top_model)
print('Model loaded1.')

for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=['mean_squared_logarithmic_error', 'accuracy'])
print('Model loaded2.')

train_datagen = ImageDataGenerator(shear_range=0.3, zoom_range=0.3, rotation_range=0.3)
print('Model loaded3.')

test_datagen = ImageDataGenerator()
print('Model loaded4.')

print('training')
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)
  

print('testing')
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='categorical',
        shuffle=False)

class_dictionary = train_generator.class_indices
sorted_class_dictionary = OrderedDict(sorted(class_dictionary.items()))
sorted_class_dictionary = sorted_class_dictionary.values()
print(sorted_class_dictionary)

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=train_generator,
        nb_val_samples=nb_train_samples)
        
model.save_weights(whole_weights)

aux = model.predict_generator(test_generator, nb_test_samples)
predictions = np.zeros((nb_test_samples, 6))

ord=[0, 1, 2, 3, 4, 5]

for n in range(6):
    i = ord[n]
    print(i)
    print(aux[:, i])
    predictions[:, n] = aux[:, i]


def get_im(path, img_width, img_height, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    resized = cv2.resize(img, (img_height, img_width))
    return resized

def load_test(img_width, img_height, color_type=1):
    print('Read test images')
    path = os.path.join(test_images_path, '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_width, img_height, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
    print(total)
    return X_test, X_test_id

X_test, test_id = load_test(img_width, img_height, color_type_global)

def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5'])
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

create_submission(predictions, test_id)
