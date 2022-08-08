from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


# Step 1: make data usable
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 2: build model

model = models.Sequential()
# CNN layers
'''
CNN layers perform better than Dense layers because CNN layers look for local patterns while
Dense layers look for global patterns. The pattenrs that CNN layers find are translation invariant
which means that the same pattern can be in a new place, but it will still recognize it. Plus,
for each convolution layer that you have, it will learn more and more hierarchical patterns. For,
example, in the letter 'x', the first convolution layer will detect 4 lines, then the second
layer will detect that these four lines form a cross of around 90 degrees, etc.

In the first convolution layer, you see the (3, 3)? That is the filter. The filter is a 3x3
matrix and it leaves a 28x28x1 picture into a 26x26x1 picture (32 filter are applied and that
leaves us with an output of 26x26x632). This filter usually moves along the picture column by 
column (stride = 1), but you can change that so it moves every 2 columns (stride = 2). This will
donwsample your image further.
'''
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

'''
What is this MaxPooling2D layer? The previous layer outputs a 26x26 map, but this layer returns a
13x13 map. Max pooling allows you to downsample your feature map (kinda like the strides). Max pooling
works like a filter, except it takes the max value in the window instead of doing any matrix
multiplication. Pooling helps you reduce the number of parameters so you don't overfit. You can
also do an average pooling instead of a max pooling.
'''
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# now we have to feed the output into a dense classifier neural net to get answers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# model.summary()

model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"tset_loss: {test_loss}, test_acc: {test_acc}")

