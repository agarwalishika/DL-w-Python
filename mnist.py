# dataset of handwritten numbers - task is to identiy the number
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

# Step 1: make the data usable
# training and testing data tuples
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

'''
# Display the fourth image in the mnist dataset
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''

# currently, the values of the images are in [0, 255] but the network expects
# the values to be in [0, 1], so we must scale it
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 2: build the model

# what?
network = models.Sequential()
# creating a dense network with 512 neurons with a relu activation function
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# this is the output layer, it has 10 neurons because there are 10 possible answers.
# we will probably take the max probability of all 10 values. normalized via softmax
network.add(layers.Dense(10, activation='softmax'))


# compilation step has three parts
#   - a loss function: to tell the model how wrong it is
#   - an optimizer: to improve the model (backpropagation)
#   - accuracy metrics: suppose, fraction of correctly identified images
network.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

# Step 3: train the model

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Step 4: evaluate
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"test acc:{test_acc}, test_loss={test_loss}")

# quick note: you probably noticed that the training accuracy is higher
# than the testing loss (look carefully) - that is overfitting :o