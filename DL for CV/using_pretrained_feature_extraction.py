import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from keras import layers
from keras import models
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


# two ways to do this: with data augmentation and without

# WITHOUT DATA AUGMENTATION

'''

# Make data usable


data_gen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(dir, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512)) # (4, 4, 512) is what conv_base returns
    labels = np.zeros(shape=(sample_count))
    generator = data_gen.flow_from_directory(
        dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    
    return features, labels

train_size = 2000
val_size = 1000
test_size = 1000

train_features, train_labels = extract_features(train_dir, train_size)
val_features, val_labels = extract_features(validation_dir, val_size)
test_features, test_labels = extract_features(test_dir, test_size)

train_features = np.reshape(train_features, (train_size, 4 * 4 * 512))
val_features = np.reshape(val_features, (val_size, 4 * 4 * 512))
test_features = np.reshape(test_features, (test_size, 4 * 4 * 512))

# create model

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=(4 * 4 * 512)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

# train model
history = model.fit(train_features, train_labels,
                    epochs=30, batch_size=20,
                    validation_data=(val_features, val_labels))

# evaluate model
eval_model(history)
'''




# WITH DATA AUGMENTATION

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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

conv_base.trainable = False # this ensures that the parameters of the convolutional base (pretrained)
                            # model are frozen and therefore cannot change. this helps reduces the
                            # number of parameters during training the classifier network
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
            
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

eval_model(history)