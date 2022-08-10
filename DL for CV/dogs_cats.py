import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from os.path import exists
from keras import models
from keras import layers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from graph_results import eval_model

model_name = 'cats_and_dogs_small_2.h5'

'''if exists(model_name):
    model = models.load_model(model_name)
else:'''
# make data usable
# we want to read the image, resize it to a 150x150, find the rgb values for each pixel
# and rescale the values from [0, 255] to [0,1]
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\train'
validation_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\validation'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20, 
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# create model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# train the model

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100, # generators will keep sampling unless you specify how many times to sample
                        # if we take 20 per epoch, and we have 100 steps, we can reach our 2k images
    epochs=30,
    validation_data=val_generator,
    validation_steps=50 # kinda like 'steps_per_epoch' but for the validation generator
)

model.save(model_name)

# assess the model
eval_model(history)
