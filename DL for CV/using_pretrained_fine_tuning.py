import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from keras import layers
from keras import models
from keras import optimizers
import numpy as np
from keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from graph_results import eval_model

conv_base = VGG16(weights='imagenet',
                  include_top=False, # whether or not to include the classifier on _top_ of the network
                  input_shape=(150, 150, 3))

base_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small'
train_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\train'
validation_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\validation'
test_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\test'

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

conv_base.trainable = False

# need to unfreeze the last few layers
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        # setting block5_conv1, conv2, conv3 and the pooling layer to 'trainable'
        set_trainable = True

    layer.trainable = set_trainable

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-5), # here, we set the learning rate :o
              loss='binary_crossentropy',
              metrics=['accuracy'])
            
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_5.3.h5')

eval_model(history)