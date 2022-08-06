from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

num_epochs = 4

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# one-hot encoding vectors into 0s and 1s. basically, we make 10k arrays for each review,
# and then if it contains a word of index x, set the index x to 1. suppose we had 5 word 
# reviews and the review [2, 3], then the vector will be [0, 0, 1, 1, 0].
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# seems like you just turn them from ints to floats
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# split your training data into a validation set of 10k (out of 25k)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = network.fit(partial_x_train,
            partial_y_train,
            epochs=num_epochs,
            batch_size=512,
            validation_data=(x_val, y_val))

test_loss, test_acc = network.evaluate(x_test, y_test)
print(f"test acc:{test_acc}, test_loss={test_loss}")

# predict the probability of a review being positive
predictions = network.predict(x_test)
print("Some predictions:")
print(predictions[0:5])
print(predictions[-5:])

# plotting training/validation loss
history_dict = history.history
loss_vals = history_dict['loss']
validation_loss_vals = history_dict['val_loss']
epochs = range(1, num_epochs + 1)

plt.plot(epochs, loss_vals, 'bo', label='Training loss')
plt.plot(epochs, validation_loss_vals, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plotting training/validation accuracy
plt.clf()
acc_vals = history_dict['accuracy']
validation_acc_vals = history_dict['val_accuracy']

plt.plot(epochs, acc_vals, 'bo', label='Training accuracy')
plt.plot(epochs, validation_acc_vals, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()